import numpy as np
import time
from scipy import special
import math
from scipy.linalg import solve_triangular


def integral_kernel_fcn(co1, co2, sigma_f=1, l=0.5):
    """Integral kernel function for STAMP - corrected implementation"""
    p0, p1, p2, p3, p4, p5, p6, p7 = [], [], [], [], [], [], [], []
    area1 = []
    area2 = []

    for i in co1:
        p0.append(i[0])
        p1.append(i[1])
        p4.append(i[2])
        p5.append(i[3])
        area1.append((i[1] - i[0]) * (i[3] - i[2]))
    for i in co2:
        p2.append(i[0])
        p3.append(i[1])
        p6.append(i[2])
        p7.append(i[3])
        area2.append((i[1] - i[0]) * (i[3] - i[2]))

    p0 = np.array([p0]).T
    p1 = np.array([p1]).T
    p2 = np.array([p2])
    p3 = np.array([p3])
    p4 = np.array([p4]).T
    p5 = np.array([p5]).T
    p6 = np.array([p6])
    p7 = np.array([p7])

    area1 = np.array([area1]).T
    area2 = np.array([area2])

    area = area1 * area2

    return covariance_cal(p0, p1, p2, p3, p4, p5, p6, p7, sigma_f, l) / area


def covariance_cal(p0, p1, p2, p3, p4, p5, p6, p7, sigma_f, l):
    """Covariance calculation for integral kernel"""
    def term_cal(a, b):
        return (b - a) * special.erf((1 / np.sqrt(2 * l**2)) * (a - b)) - np.exp(
            -0.5 * (a - b) ** 2 / l**2
        ) / np.sqrt(0.5 * np.pi / l**2)

    return (
        0.5
        * np.pi
        * l**2
        * sigma_f**2
        * (term_cal(p1, p3) - term_cal(p1, p2) - term_cal(p0, p3) + term_cal(p0, p2))
        * (term_cal(p5, p7) - term_cal(p5, p6) - term_cal(p4, p7) + term_cal(p4, p6))
    )


def matrix_inverse(m):
    """Matrix inverse with numerical stability"""
    d = len(m)
    m = 0.5 * np.add(m, np.transpose(m)) + 1e-8 * np.eye(d)
    m = np.transpose(np.linalg.cholesky(m))
    m = np.linalg.inv(m)
    return np.dot(m, np.transpose(m))


class STAMP_FR_GPF_IK:
    """Fixed Resolution Gaussian Process Fusion with Integral Kernels for STAMP"""
    
    def __init__(self, x_boundary=1.0, y_boundary=1.0, x_resolution=32, y_resolution=32, 
                 f_prior=0.0, v_prior=1.0, l=0.1, elevated_regions=None, flight_altitude=0.1, temporal_alpha=0.90):
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.f_prior = f_prior
        self.v_prior = v_prior
        self.l = l
        self.elevated_regions = elevated_regions if elevated_regions is not None else []
        self.flight_altitude = flight_altitude
        self.temporal_alpha = temporal_alpha

        # Map setting
        self.x_gsd = self.x_boundary / self.x_resolution
        self.y_gsd = self.y_boundary / self.y_resolution
        self.state_vector = None  # cell center list
        self.state_value = None  # cell value list
        self.grid_corners = None  # grid cell corners for integral kernel
        self.covariance = None
        self.exploration_mask = None
        self.mapping_time = 0
        # Mask for grids inside elevated regions (will be set during initialization)
        self.obstructed_grids_mask = None

        self.init_f_map()
        self.init_v_map()

        # --- NEW: store structural GP prior (fixed spatial kernel) ---
        self.prior_mean_map = self.state_value.copy()      # μ0

    def _point_in_convex_polygon(self, point, vertices):
        """
        Check if a point is inside a convex polygon.
        Uses cross product method - works for both clockwise and counter-clockwise vertex order.
        For a convex polygon, a point is inside if it's on the same side of all edges.
        """
        if len(vertices) < 3:
            return False
        vertices = np.array(vertices)
        point = np.array(point)
        n = len(vertices)
        
        # Collect cross products for all edges
        cross_products = []
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            # Vector from v1 to v2
            edge_vec = v2 - v1
            # Vector from v1 to point
            point_vec = point - v1
            # Cross product (z-component): positive if point is to the left of edge
            cross = edge_vec[0] * point_vec[1] - edge_vec[1] * point_vec[0]
            cross_products.append(cross)
        
        # For a convex polygon, point is inside if all cross products have the same sign
        # (all positive or all negative, depending on vertex winding order)
        if len(cross_products) == 0:
            return False
        
        # Check if all cross products have the same sign
        all_positive = all(cp > 0 for cp in cross_products)
        all_negative = all(cp < 0 for cp in cross_products)
        
        # Also allow points exactly on edges (cross == 0)
        if all_positive or all_negative:
            return True
        
        # Check if all are non-negative or all are non-positive (including zero)
        all_non_negative = all(cp >= 0 for cp in cross_products)
        all_non_positive = all(cp <= 0 for cp in cross_products)
        
        return all_non_negative or all_non_positive
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        Check if two line segments (p1-p2 and p3-p4) intersect.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _line_segment_intersects_polygon(self, p1, p2, vertices):
        """
        Check if a line segment from p1 to p2 intersects a convex polygon.
        Returns True if the segment intersects or is inside the polygon.
        """
        vertices = np.array(vertices)
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        # Check if either endpoint is inside the polygon
        if self._point_in_convex_polygon(p1, vertices) or self._point_in_convex_polygon(p2, vertices):
            return True
        
        # Check if segment intersects any edge of the polygon
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            if self._line_segments_intersect(p1, p2, v1, v2):
                return True
        return False
    
    def _is_grid_center_in_region(self, grid_center):
        """Check if a grid center is inside any elevated region"""
        for region in self.elevated_regions:
            if self._point_in_convex_polygon(grid_center, region['vertices']):
                return True
        return False
    
    def _is_grid_obstructed(self, grid_center, agent_pos, agent_altitude):
        """
        Check if a grid is obstructed by elevated regions.
        A grid is obstructed if the line from grid center to agent position
        intersects with any elevated region that is above the agent's altitude.
        """
        for region in self.elevated_regions:
            # Only check regions above agent altitude
            if region['height'] > agent_altitude:
                if self._line_segment_intersects_polygon(grid_center, agent_pos, region['vertices']):
                    return True
        return False
    
    def _reset_region_grids(self):
        """Reset grids inside elevated regions after each update."""
        if self.obstructed_grids_mask is None:
            return
        idx = np.where(self.obstructed_grids_mask)[0]
        if idx.size == 0:
            return

        self.state_value[idx] =0.0
        self.state_var[idx] = 0.0
        self._sync_covariance_diag()


    def init_f_map(self):
        """Initialize field map"""
        # Create grid cell centers
        x = np.arange(0, self.x_boundary, self.x_gsd) + 0.5 * self.x_gsd
        y = np.arange(0, self.y_boundary, self.y_gsd) + 0.5 * self.y_gsd
        xx, yy = np.meshgrid(x, y)
        self.state_vector = np.c_[xx.ravel(), yy.ravel()]
        self.state_value = self.f_prior * np.ones(len(self.state_vector))
        self.exploration_mask = np.zeros(len(self.state_vector), dtype=bool)
        
        # Initialize mask for grids inside elevated regions
        self.obstructed_grids_mask = np.zeros(len(self.state_vector), dtype=bool)
        
        # Set grids inside elevated regions to -1
        for i, center in enumerate(self.state_vector):
            if self._is_grid_center_in_region(center):
                self.state_value[i] = 0.0
                self.obstructed_grids_mask[i] = True
        
        # Create grid cell corners for integral kernel computation
        self.grid_corners = []
        for i, center in enumerate(self.state_vector):
            x_min = center[0] - 0.5 * self.x_gsd
            x_max = center[0] + 0.5 * self.x_gsd
            y_min = center[1] - 0.5 * self.y_gsd
            y_max = center[1] + 0.5 * self.y_gsd
            self.grid_corners.append([x_min, x_max, y_min, y_max])
        
        self.grid_corners = np.array(self.grid_corners)

    def init_v_map(self):
        """Initialize variance model: prior diag + fixed correlation template."""
        # Full prior covariance (kernel-induced)
        K = integral_kernel_fcn(self.grid_corners, self.grid_corners, self.v_prior, self.l)
        K = 0.5 * (K + K.T) + 1e-10 * np.eye(K.shape[0])

        # Zero-out obstructed cells in the prior
        if self.obstructed_grids_mask is not None and np.any(self.obstructed_grids_mask):
            idx = np.where(self.obstructed_grids_mask)[0]
            K[idx, :] = 0.0
            K[:, idx] = 0.0
            K[idx, idx] = 0.0

        # Store prior diagonal (needed for process update)
        self.prior_diag = np.diag(K).copy()

        # Build fixed correlation template C = D^{-1/2} K D^{-1/2}
        d = np.sqrt(np.maximum(self.prior_diag, 1e-12))
        d_safe = d.copy()
        d_safe[d_safe < 1e-12] = 1.0  # avoid divide-by-zero

        C = K / (d_safe[:, None] * d_safe[None, :])
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)

        # Make obstructed rows/cols safe (no propagation)
        if self.obstructed_grids_mask is not None and np.any(self.obstructed_grids_mask):
            idx = np.where(self.obstructed_grids_mask)[0]
            C[idx, :] = 0.0
            C[:, idx] = 0.0
            C[idx, idx] = 1.0

        self.corr = C

        # Current marginal variances (diag of posterior)
        self.state_var = self.prior_diag.copy()

        # Compatibility 2D covariance (diagonal-only)
        self.covariance = np.zeros_like(K)
        self._sync_covariance_diag()



    def take_measurement(self, agent_pos, target_positions, agent_heading=None, 
                        camera_pitch=None, camera_hfov=None, camera_vfov=None, 
                        flight_altitude=None, fov_size=0.1):
        """
        Take measurement from agent position with realistic camera model
        
        Args:
            agent_pos: Agent position [x, y] in normalized coordinates
            target_positions: List of target positions [[x1, y1], [x2, y2], ...]
            agent_heading: Agent heading in radians (0 = +x direction, counterclockwise)
            camera_pitch: Camera pitch angle in degrees (0-90, 0=horizontal, 90=vertical down)
            camera_hfov: Horizontal field of view in degrees
            camera_vfov: Vertical field of view in degrees
            flight_altitude: Flight altitude in normalized units
            fov_size: Legacy parameter (deprecated, kept for backward compatibility)
        
        Returns:
            m: Measurement values (1.0 if target detected, -1.0 otherwise)
            Hm: List of cell indices in field of view
            distances: Array of distances from agent to each cell center
            targets_observed: Boolean array indicating which targets are observed [len(target_positions)]
        """
        # Use provided parameters or fall back to legacy fov_size
        if camera_pitch is None or camera_hfov is None or camera_vfov is None or flight_altitude is None:
            # Legacy mode: use simple square FOV
            x_min = agent_pos[0] - fov_size/2
            x_max = agent_pos[0] + fov_size/2
            y_min = agent_pos[1] - fov_size/2
            y_max = agent_pos[1] + fov_size/2
            fov = np.array([x_min, x_max, y_min, y_max])
            query_list = self.state_query(fov)
        else:
            # Realistic camera model
            query_list = self._get_cells_in_camera_fov(
                agent_pos, agent_heading, camera_pitch, camera_hfov, 
                camera_vfov, flight_altitude
            )
        
        # Use provided flight_altitude or fall back to instance default
        agent_altitude = flight_altitude if flight_altitude is not None else self.flight_altitude
        
        # Process measurement (binary: 1 if target in cell, -1 otherwise)
        # Filter out grids inside elevated regions and obstructed grids
        m = []
        Hm = []
        distances = []  # Distance from agent to each cell center
        m_notarget_detected = []
        Hm_notarget_detected = []
        distances_notarget_detected = []  # Distance from agent to each cell center
        targets_observed = np.zeros(len(target_positions), dtype=bool)  # Track which targets are observed
        
        for cell_idx in query_list:
            # Skip grids inside elevated regions
            if self.obstructed_grids_mask[cell_idx]:
                continue
            
            cell_center = self.state_vector[cell_idx]
            
            # Skip grids obstructed by elevated regions (line-of-sight check)
            if self._is_grid_obstructed(cell_center, agent_pos, agent_altitude):
                continue
            
            cell_x_min = cell_center[0] - 0.5 * self.x_gsd
            cell_x_max = cell_center[0] + 0.5 * self.x_gsd
            cell_y_min = cell_center[1] - 0.5 * self.y_gsd
            cell_y_max = cell_center[1] + 0.5 * self.y_gsd
            # Check if any target is within this cell
            target_detected = 0.0
            for t_idx, target_pos in enumerate(target_positions):
                # Check if target is within cell boundaries
                if (cell_x_min <= target_pos[0] <= cell_x_max and 
                    cell_y_min <= target_pos[1] <= cell_y_max):
                    target_detected = 2.0
                    targets_observed[t_idx] = True  # Mark this target as observed
                    break
            if target_detected == 0.0:
                m_notarget_detected.append(target_detected)
                Hm_notarget_detected.append(cell_idx)
                dist = np.linalg.norm(cell_center - agent_pos)
                distances_notarget_detected.append(dist)
            else:
                m.append(target_detected)
                Hm.append(cell_idx)
                # Compute distance from agent to cell center
                dist = np.linalg.norm(cell_center - agent_pos)
                distances.append(dist)

        # Diagnostic only (read by the opt-in eval counters in env.py): number of
        # negative footprint cells discarded below because a target was in view.
        self.last_negatives_dropped = len(m_notarget_detected) if len(m) > 0 else 0
        if len(m) > 0:
            return np.array(m), Hm, np.array(distances), targets_observed
        else:
            return np.array(m_notarget_detected), Hm_notarget_detected, np.array(distances_notarget_detected), targets_observed
    
    def _get_cells_in_camera_fov(self, agent_pos, agent_heading, camera_pitch, 
                                  camera_hfov, camera_vfov, flight_altitude):
        """
        Calculate which grid cells are in the camera's field of view
        
        Args:
            agent_pos: Agent position [x, y]
            agent_heading: Agent heading in radians (0 = +x direction)
            camera_pitch: Camera pitch angle in degrees (0-90)
            camera_hfov: Horizontal FOV in degrees
            camera_vfov: Vertical FOV in degrees
            flight_altitude: Flight altitude in normalized units
        
        Returns:
            List of cell indices in FOV
        """
        # Convert angles to radians
        pitch_rad = np.deg2rad(camera_pitch)
        hfov_rad = np.deg2rad(camera_hfov)
        vfov_rad = np.deg2rad(camera_vfov)
        
        # Calculate ground coverage from camera geometry
        # Half-angles
        half_hfov = hfov_rad / 2.0
        half_vfov = vfov_rad / 2.0
        
        if pitch_rad < np.pi/2 - 1e-6:  # Not looking straight down
            # Calculate distances to near and far edges of vertical FOV on ground
            pitch_min = max(0.0, pitch_rad - half_vfov)
            pitch_max = min(np.pi/2 - 1e-6, pitch_rad + half_vfov)
            
            # Distance along ground from agent to near and far edges
            near_dist = flight_altitude * np.tan(pitch_min) if pitch_min > 0 else 0.0
            far_dist = flight_altitude * np.tan(pitch_max) if pitch_max < np.pi/2 else np.inf
            
            # For each distance, calculate the width of FOV at that distance
            # The width depends on the horizontal FOV and the distance from camera
            # At distance d from agent, the width is: 2 * d * tan(half_hfov) / cos(pitch)
            # But we need to account for the 3D geometry properly
            
            # Simplified: use average pitch angle for width calculation
            avg_pitch = pitch_rad
            near_width = 2 * near_dist * np.tan(half_hfov) / np.cos(avg_pitch) if near_dist > 0 else 0.0
            far_width = 2 * far_dist * np.tan(half_hfov) / np.cos(avg_pitch) if far_dist < np.inf else np.inf
            
            # Create FOV polygon vertices in camera frame (forward = +x, right = +y)
            vertices_camera = []
            if near_dist > 1e-6:
                # Near edge vertices
                vertices_camera.append([near_dist, -near_width/2])
                vertices_camera.append([near_dist, near_width/2])
            if far_dist < 1e6:  # Use finite check instead of np.inf
                # Far edge vertices (order matters for polygon)
                vertices_camera.append([far_dist, far_width/2])
                vertices_camera.append([far_dist, -far_width/2])
            
            if len(vertices_camera) < 3:
                # If pitch is too shallow or FOV doesn't intersect ground properly
                # Fall back to simple square FOV
                avg_dist = (near_dist + far_dist) / 2.0 if far_dist < 1e6 else near_dist * 2.0
                return self._get_cells_in_simple_fov(agent_pos, max(avg_dist * np.tan(half_hfov), flight_altitude * np.tan(half_hfov)))
            
            # Ensure vertices form a proper polygon (close the loop)
            if len(vertices_camera) == 4:
                # Reorder to form a proper quadrilateral
                # Order: near-left, near-right, far-right, far-left
                vertices_camera = [
                    vertices_camera[0],  # near-left
                    vertices_camera[1],  # near-right
                    vertices_camera[2],  # far-right
                    vertices_camera[3]   # far-left
                ]
            
            # Rotate vertices by heading and translate to agent position
            cos_h = np.cos(agent_heading)
            sin_h = np.sin(agent_heading)
            rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            
            vertices_world = []
            for v in vertices_camera:
                v_rotated = rotation_matrix @ np.array(v)
                v_world = v_rotated + agent_pos
                vertices_world.append(v_world)
            
            # Find all cells whose centers are inside the FOV polygon
            query_list = []
            for cell_idx in range(len(self.state_vector)):
                cell_center = self.state_vector[cell_idx]
                if self._point_in_polygon(cell_center, vertices_world):
                    query_list.append(cell_idx)
            
            return query_list
        else:
            # Looking straight down (pitch = 90 degrees)
            # FOV is a rectangle on the ground
            half_width = flight_altitude * np.tan(hfov_rad / 2.0)
            half_height = flight_altitude * np.tan(vfov_rad / 2.0)
            
            # Rotate the rectangle by heading
            cos_h = np.cos(agent_heading)
            sin_h = np.sin(agent_heading)
            
            # Corners of rectangle in camera frame
            corners_camera = np.array([
                [-half_height, -half_width],
                [-half_height, half_width],
                [half_height, half_width],
                [half_height, -half_width]
            ])
            
            # Rotate and translate
            rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            corners_world = (rotation_matrix @ corners_camera.T).T + agent_pos
            
            # Find cells in rotated rectangle
            query_list = []
            for cell_idx in range(len(self.state_vector)):
                cell_center = self.state_vector[cell_idx]
                if self._point_in_polygon(cell_center, corners_world):
                    query_list.append(cell_idx)
            
            return query_list
    
    def _get_cells_in_simple_fov(self, agent_pos, fov_radius):
        """Fallback: simple square FOV"""
        x_min = agent_pos[0] - fov_radius
        x_max = agent_pos[0] + fov_radius
        y_min = agent_pos[1] - fov_radius
        y_max = agent_pos[1] + fov_radius
        fov = np.array([x_min, x_max, y_min, y_max])
        return self.state_query(fov)
    
    def _point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using ray casting algorithm
        
        Args:
            point: Point [x, y]
            polygon: List of polygon vertices [[x1, y1], [x2, y2], ...]
        
        Returns:
            True if point is inside polygon
        """
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def state_query(self, fov):
        """Find grid cells that overlap with field of view"""
        query_list = []
        
        for i, center in enumerate(self.state_vector):
            x_min = center[0] - 0.5 * self.x_gsd
            x_max = center[0] + 0.5 * self.x_gsd
            y_min = center[1] - 0.5 * self.y_gsd
            y_max = center[1] + 0.5 * self.y_gsd
            
            # Check overlap with FoV
            if (x_max >= fov[0] and x_min <= fov[1] and 
                y_max >= fov[2] and y_min <= fov[3]):
                query_list.append(i)
        
        return query_list

    def update_map(
        self,
        m,
        Hm,
        noise_std=0.1,
        regularization=1e-6,
        max_condition_number=1e8,
        distances=None,
        noise_distance_scale=1.0,
        pos_noise_scale=1.0,  # <1 trust positives more
        neg_noise_scale=1.0,  # >1 trust negatives less
    ):
        """
        Measurement update using implicit covariance:
        P ≈ diag(sqrt(p)) * C * diag(sqrt(p)),
        where C is fixed correlation matrix, and p is diag variance.

        Keeps self.covariance as diagonal-only for compatibility with step().
        """
        if len(Hm) == 0:
            return

        # Filter out obstructed grids (keep distances aligned)
        valid_idx, valid_m, valid_dist = [], [], []
        for i, cell_idx in enumerate(Hm):
            if not self.obstructed_grids_mask[cell_idx]:
                valid_idx.append(cell_idx)
                valid_m.append(m[i])
                if distances is not None and i < len(distances):
                    valid_dist.append(distances[i])

        if len(valid_idx) == 0:
            return

        H = np.asarray(valid_idx, dtype=int)
        z = np.asarray(valid_m, dtype=float).reshape(-1)

        # Distance-based noise
        if distances is not None and len(valid_dist) == len(z):
            dist = np.asarray(valid_dist, dtype=float)
            noise_stds = noise_std * (1.0 + noise_distance_scale * dist)
        else:
            noise_stds = noise_std * np.ones_like(z)

        # Optional: different confidence for positive vs negative measurements
        # (works for your -1/10 scheme: positive if z > 0)
        pos_mask = z > 0.0
        noise_stds = noise_stds * np.where(pos_mask, pos_noise_scale, neg_noise_scale)
        r = noise_stds**2  # variances

        # Current mean/diag variance
        mu = self.state_value
        p = np.maximum(self.state_var, 1e-12)
        sqrt_p = np.sqrt(p)
        sqrt_p_H = sqrt_p[H]

        # Build P_{:,H} and P_{H,H} implicitly from corr + diag variances
        P_xH = (sqrt_p[:, None] * self.corr[:, H]) * sqrt_p_H[None, :]          # (V, m)
        C_HH = self.corr[np.ix_(H, H)]
        P_HH = (sqrt_p_H[:, None] * C_HH) * sqrt_p_H[None, :]                  # (m, m)

        # Innovation covariance Sigma = P_HH + R + reg*I
        Sigma = 0.5 * (P_HH + P_HH.T)
        Sigma[np.diag_indices_from(Sigma)] += r + regularization

        # Condition number guard (m ~ 100 so this is cheap)
        try:
            eigvals = np.linalg.eigvalsh(Sigma)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
            cond = max_eig / min_eig if min_eig > 0 else np.inf
            if cond > max_condition_number:
                target_min = max_eig / max_condition_number
                extra = max(0.0, target_min - min_eig)
                Sigma[np.diag_indices_from(Sigma)] += extra
        except Exception:
            pass

        # Cholesky solve
        try:
            L = np.linalg.cholesky(Sigma)  # lower
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(Sigma.shape[0]))

        # Innovation
        nu = z - mu[H]

        # gain = P_xH * Sigma^{-1} via triangular solves
        tmp = solve_triangular(L, P_xH.T, lower=True)          # (m, V) = L^{-1} P^T
        gain_T = solve_triangular(L.T, tmp, lower=False)       # (m, V) = Sigma^{-1} P^T
        gain = gain_T.T                                        # (V, m)

        # Mean update
        mu_post = mu + gain @ nu

        # Diagonal variance update: p_post = p - diag(P_xH Sigma^{-1} P_Hx)
        # With W = P_xH L^{-T} = tmp.T, diag reduction = row-wise sum(W^2)
        W = tmp.T                                              # (V, m)
        p_post = p - np.sum(W * W, axis=1)
        p_post = np.maximum(p_post, 1e-12)

        self.state_value = mu_post
        self.state_var = p_post
        self.exploration_mask[H] = True

        # Keep compatibility covariance updated + enforce obstacles
        self._sync_covariance_diag()
        self._reset_region_grids()



    def get_map_grid(self):
        """Get map as 2D grid for visualization"""
        return self.state_value.reshape(self.y_resolution, self.x_resolution)
    
    def get_variance_grid(self):
        """Get variance map as 2D grid for visualization"""
        return self.state_var.reshape(self.y_resolution, self.x_resolution)

    def get_cell_coordinates(self):
        """Get cell coordinates as 2D grids"""
        x = np.arange(0, self.x_boundary, self.x_gsd) + 0.5 * self.x_gsd
        y = np.arange(0, self.y_boundary, self.y_gsd) + 0.5 * self.y_gsd
        return np.meshgrid(x, y)
    
    def _points_to_grid_indices(self, points):
        """
        Convert point coordinates to grid cell indices efficiently.
        Uses direct calculation based on grid structure instead of distance search.
        
        Args:
            points: Array of shape (N, 2) with [x, y] coordinates
            
        Returns:
            indices: Array of shape (N,) with grid cell indices for each point
        """
        if len(points) == 0:
            return np.array([], dtype=int)
        
        points = np.asarray(points)
        # Grid cells are centered at: 0.5*gsd, 1.5*gsd, 2.5*gsd, ...
        # For a point (x, y), find which cell it belongs to
        # x_idx = floor(x / x_gsd), clamped to [0, x_resolution-1]
        # y_idx = floor(y / y_gsd), clamped to [0, y_resolution-1]
        # idx = y_idx * x_resolution + x_idx
        
        x_indices = np.floor(points[:, 0] / self.x_gsd).astype(int)
        y_indices = np.floor(points[:, 1] / self.y_gsd).astype(int)
        
        # Clamp to valid grid bounds
        x_indices = np.clip(x_indices, 0, self.x_resolution - 1)
        y_indices = np.clip(y_indices, 0, self.y_resolution - 1)
        
        # Convert to linear indices (row-major order: y_idx * x_resolution + x_idx)
        indices = y_indices * self.x_resolution + x_indices
        
        return indices
    
    def get_field_values_and_variances_at_points(self, points, num_process_updates=0, alpha=None):
        if len(points) == 0:
            return np.array([]), np.array([])

        indices = self._points_to_grid_indices(points)

        if num_process_updates == 0:
            values = self.state_value[indices]
            variances = self.state_var[indices]
            return values, variances

        if alpha is None:
            alpha = self.temporal_alpha

        future_state = self.state_value.copy()
        future_var = self.state_var.copy()

        for _ in range(num_process_updates):
            future_state = self.prior_mean_map + alpha * (future_state - self.prior_mean_map)
            future_var = (alpha**2) * future_var + (1.0 - alpha**2) * self.prior_diag

        # enforce obstacles
        if self.obstructed_grids_mask is not None and np.any(self.obstructed_grids_mask):
            idx = np.where(self.obstructed_grids_mask)[0]
            future_state[idx] = 0.0
            future_var[idx] = 0.0

        return future_state[indices], future_var[indices]

    
    def get_field_value_at_points(self, points):
        """
        Get field value at given points (current state).
        Wrapper for backward compatibility.
        """
        values, _ = self.get_field_values_and_variances_at_points(points, num_process_updates=0)
        return values
    
    def get_field_variance_at_points(self, points):
        """
        Get field variance at given points (current state).
        Wrapper for backward compatibility.
        """
        _, variances = self.get_field_values_and_variances_at_points(points, num_process_updates=0)
        return variances
    
    def get_future_field_value_at_points(self, points, num_process_updates=2, alpha=None):
        """
        Get predicted field value at given points after num_process_updates process updates.
        Wrapper for backward compatibility.
        """
        values, _ = self.get_field_values_and_variances_at_points(points, num_process_updates, alpha)
        return values
    
    def get_future_field_variance_at_points(self, points, num_process_updates=2, alpha=None):
        """
        Get predicted field variance at given points after num_process_updates process updates.
        Wrapper for backward compatibility.
        """
        _, variances = self.get_field_values_and_variances_at_points(points, num_process_updates, alpha)
        return variances

    def process_update(self, alpha=None):
        """
        Diagonal spatio-temporal process update:
        mu^- = mu0 + alpha*(mu - mu0)
        p^-  = alpha^2 p + (1-alpha^2) prior_diag
        """
        if alpha is None:
            alpha = self.temporal_alpha

        # Mean prediction
        self.state_value = self.prior_mean_map + alpha * (self.state_value - self.prior_mean_map)

        # Variance prediction (diagonal only)
        self.state_var = (alpha**2) * self.state_var + (1.0 - alpha**2) * self.prior_diag
        self.state_var = np.maximum(self.state_var, 1e-12)

        # Keep compatibility matrix updated
        self._sync_covariance_diag()

        # Enforce obstacle cells
        self._reset_region_grids()



    def update_map_spatial_temporal(
        self,
        m,
        Hm,
        noise_std=0.1,
        regularization=1e-6,
        max_condition_number=1e8,
        alpha=None,
        distances=None,
        noise_distance_scale=1.0,
        pos_noise_scale=1.0,
        neg_noise_scale=1.0,
    ):
        self.process_update(alpha=alpha)
        self.update_map(
            m, Hm,
            noise_std=noise_std,
            regularization=regularization,
            max_condition_number=max_condition_number,
            distances=distances,
            noise_distance_scale=noise_distance_scale,
            pos_noise_scale=pos_noise_scale,
            neg_noise_scale=neg_noise_scale,
        )


    def get_belief_packet(self):
        """
        What to broadcast: mean + marginal variances.
        Much smaller than sending VxV.
        """
        return {
            "mu": self.state_value.copy(),
            "var": self.state_var.copy(),
        }

    def fuse_belief_ci(self, mu_other, var_other, omega=0.5):
        """
        Diagonal Covariance Intersection (CI) fusion (per cell).
        Conservative under unknown cross-correlation (intermittent comm).

        omega: scalar weight in [0,1], or the string 'trace' to pick the
        scalar weight minimizing tr(P_c) by 1D grid search per exchange.
        """
        mu1 = self.state_value
        v1 = np.maximum(self.state_var, 1e-12)

        mu2 = np.asarray(mu_other, dtype=float)
        v2 = np.maximum(np.asarray(var_other, dtype=float), 1e-12)

        inv1 = 1.0 / v1
        inv2 = 1.0 / v2

        if isinstance(omega, str) and omega == "trace":
            candidates = np.linspace(0.05, 0.95, 19)
            traces = [np.sum(1.0 / (w * inv1 + (1.0 - w) * inv2)) for w in candidates]
            omega = float(candidates[int(np.argmin(traces))])

        inv = omega * inv1 + (1.0 - omega) * inv2
        v = 1.0 / inv
        mu = v * (omega * inv1 * mu1 + (1.0 - omega) * inv2 * mu2)

        self.state_value = mu
        self.state_var = v

        self._reset_region_grids()

    def _sync_covariance_diag(self):
        """Keep a 2D covariance matrix for compatibility, but diagonal-only."""
        if self.covariance is None or self.covariance.ndim != 2:
            V = len(self.state_vector)
            self.covariance = np.zeros((V, V), dtype=float)
        np.fill_diagonal(self.covariance, self.state_var)
