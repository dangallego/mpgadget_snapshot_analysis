import bigfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from nbodykit.lab import BigFileCatalog
from scipy.spatial import distance


class SnapshotData:
    def __init__(self, snapshot_path):
        """
        Initializes the SnapshotData object with the path to the snapshot directory.
        The particle data is not loaded until one of the specific load methods is called.

        Parameters:
            snapshot_path (str): The path to the snapshot directory containing particle data.
        """
        self.snapshot_path = snapshot_path
        self.data = BigFileCatalog(snapshot_path, dataset='1/', header='Header')
        self.positions = None
        self.group_ids = None
        self.velocities = None
        self.masses = None
        self.potentials = None
        self.particle_ids = None

    def load_positions(self):
        """
        Loads and stores the position data from the snapshot if not already loaded.
        Converts positions from kpc/h to Mpc/h for better usability.
        """
        if self.positions is None:
            self.positions = self.data['Position'].compute() / 1000  # Convert from kpc/h to Mpc/h

    def load_group_ids(self):
        """
        Loads and stores the group ID data from the snapshot if not already loaded.
        """
        if self.group_ids is None:
            self.group_ids = self.data['GroupID'].compute()

    def load_velocities(self):
        """
        Loads and stores the particle velocity data from the snapshot if not already loaded.
        """
        if self.velocities is None:
            self.velocities = self.data['Velocity'].compute()

    def load_masses(self):
        """
        Loads and stores the particle velocity data from the snapshot if not already loaded.
        """
        if self.masses is None:
            self.masses = self.data['Mass'].compute()

    def load_potentials(self):
        """
        Loads and stores the particle velocity data from the snapshot if not already loaded.
        """
        if self.potentials is None:
            self.potentials = self.data['Potential'].compute()

    def load_particle_ids(self):
        """
        Loads and stores the particle velocity data from the snapshot if not already loaded.
        """
        if self.particle_ids is None:
            self.particle_ids = self.data['ID'].compute()

    def get_all_unique_ids(self):
        """
        Prints total number of unique IDs

        Parameters:
            target_ids (list of int): List of group IDs for which particles are to be filtered.
        """
        self.load_group_ids()

        # Find unique group IDs
        unique_ids, id_counts = np.unique(self.group_ids, return_counts=True)
        # Count how many group IDs exist
        num_unique_ids = len(id_counts)
        print("Unique group IDs: ", unique_ids)
        print("Total number of group IDs: ", num_unique_ids) # This is expected and given in the PIG Header attribute file

    def get_N_largest_ids(self, N=3):
        """
        Prints N number of largest group IDs as well as how many particles correspond to each ID.
        """
        self.load_group_ids()
        
        # Find unique group IDs
        unique_ids, id_counts = np.unique(self.group_ids, return_counts=True)
        
        # Get indices of the counts sorted in descending order
        sorted_indices = np.argsort(id_counts)[::-1]
        
        # Get indices of top N group IDs with the highest counts
        top_N_ids = unique_ids[sorted_indices[:N]]
        top_N_counts = id_counts[sorted_indices[:N]]
        
        # Print results
        print("Top N Group IDs with highest number of particles: ")
        for g_id, count in zip(top_N_ids, top_N_counts):
            print(f"Group ID: {g_id}, Count: {count}")

    def filter_particles_by_group(self, target_ids=None):
        """
        Filters and returns particle coordinates for specified group IDs. Ensures position data is loaded.

        Parameters:
            target_ids (list of int, optional): List of group IDs for which particles are to be filtered. If none, filters for all group IDs.

        Returns:
            dict: A dictionary with group IDs as keys and corresponding particle coordinates as values.
        """
        self.load_positions()
        self.load_positions()

        if target_ids is None:
            target_ids = np.unique(self.group_ids[self.group_ids!=4294967295])

        filtered_particles = {} # initialize dictionary of group ID an particle coordinates
        for group_id in target_ids:
            mask = self.group_ids == group_id
            filtered_particles[group_id] = self.positions[mask]
        return filtered_particles

    def plot_particle_groups(self, target_ids):
        """
        Plots particles for the specified group IDs in a 2D scatter plot.

        Parameters:
            target_ids (list or array-like): group IDs to consider for processing. 
        """
        self.load_positions()
        particle_dict = self.filter_particles_by_group(target_ids)
        plt.figure(figsize=(12, 10))
        cmap = cm.get_cmap('viridis', len(target_ids))
        color_idx = np.linspace(0, 1, len(target_ids))

        for idx, group_id in enumerate(target_ids):
            x, y = particle_dict[group_id][:, 0], particle_dict[group_id][:, 1]
            plt.scatter(x, y, color=cmap(color_idx[idx]), s=5, alpha=0.5, label=f'Group ID: {group_id}', edgecolor='none')

        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ticks=np.linspace(0, 1, len(target_ids)), format=plt.FuncFormatter(lambda val, loc: target_ids[loc]))
        plt.xlabel('X (Mpc/h)')
        plt.ylabel('Y (Mpc/h)')
        plt.title('Particle positions by Group ID')
        plt.grid(True)
        #plt.legend(title='Group IDs')
        plt.show()

    def plot_3D_particle_groups(self, target_ids, legend=False):
        """
        Plots multiple particle groups in 2D.

        Parameters:
            target_ids (list or array-like): group IDs to consider for processing.
        """
        particle_dict = self.filter_particles_by_group(target_ids)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111,projection='3d')
        # Define list of colors for the plots
        colors = plt.cm.jet(np.linspace(0,1,len(target_ids)))  # Get colormap with as many colors as groups
        
        for group_id, color in zip(target_ids, colors):
            # Extract positions
            x, y, z = particle_dict[group_id][:,0], particle_dict[group_id][:,1], particle_dict[group_id][:,2]
            ax.scatter(x,y,z, c=[color], s=10, alpha=0.5, label=f'Group ID: {group_id}')

        ax.set_xlabel('X (Mpc/h)')
        ax.set_ylabel('Y (Mpc/h)')
        ax.set_zlabel('Z (Mpc/h)')
        ax.set_title('Particle positions by Group ID')
        if legend:
            ax.legend(title='Group IDs', loc='upper left', bbox_to_anchor=(1.05,1))
        plt.show()

    def plot_all_groups(self):
        """
        Plots all particle groups in 2D.
        """
        self.load_group_ids()
        self.load_positions()
        plt.figure(figsize=(12,10))

        # Find unique group IDs and prepare for coloring
        unique_ids = np.unique(self.group_ids)
        unique_ids = unique_ids[unique_ids != 4294967295]  # This group ID confusingly corresponds to particles that are NOT members of a group
        color_map = cm.get_cmap('viridis', len(unique_ids))

        x,y = self.positions[:,0], self.positions[:,1] 
        for i, g_id in enumerate(unique_ids):
            mask = self.group_ids == g_id
            plt.scatter(x[mask], y[mask], color=color_map(i), s=0.5, alpha=0.5, label=f'Group {g_id}')      
        plt.xlabel('X (Mpc/h)')
        plt.ylabel('Y (Mpc/h)')
        plt.title('Particle positions for All Groups')
        plt.legend(loc='upper left', bbox_to_anchor=(0.95,1))
        plt.show()

    def get_significant_groups(self, sigma=1):
        """
        Computes all group IDs, their counts (number of particles in the group), and then filters out the 
        group IDs where the count is more than 'sigma' standard deviation above the mean.

        Parameters:
            sigma (int): Optional, number of 'sigma' standard deviations by which to filter groups. 

        Returns: 
            significant_ids (array_like): Group IDs that contain 'sigma' number of particles above the mean. 
            siificant_counts (array_like): Counts (number of particles) belonging to the group IDs that satisfy the above requirement. 
        """
        self.load_group_ids()

        mask = self.group_ids != 4294967295  # do not include group corresponding to particles NOT part of a group
        ids = self.group_ids[mask]

        # Calculate unique group IDs and counts
        unique_ids,id_counts = np.unique(ids, return_counts=True)

        # Calculate mean and standard deviation
        mean_counts = np.mean(id_counts)
        std_counts = np.std(id_counts)

        # Identify group IDs with counts more than sigma above mean
        significant_mask = id_counts > (mean_counts + sigma*std_counts)
        significant_ids = unique_ids[significant_mask]
        signficant_counts = id_counts[significant_mask]

        significant_ids = unique_ids[id_counts > (mean_counts + sigma*std_counts)]

        return significant_ids, signficant_counts
    
    def find_closest_groups(self, points_of_interest, threshold=5):
        """
        Finds the closest groups to specified points of interest based on particle proximity.

        Parameters:
            points_of_interest (list of tuples): Points to find closest groups to, e.g., [(x1, y1), (x2, y2)].
            threshold (int): How many of the closest particles to consider for determining the group ID.
            
        Returns:
            dict: A dictionary with points as keys and the most frequent group ID for that point as values.
        """
        self.load_positions()
        self.load_group_ids()

        result = {}
        for point in points_of_interest:
            all_distances = []
            for group_id, particles in self.filter_particles_by_group().items():
                # Calculate distances from each particle in this group to the point
                for particle in particles:
                    dist = np.linalg.norm(particle[:2] - np.array(point))  # Assume particle[:2] are x, y coordinates
                    all_distances.append((dist, group_id))

            # Sort distances and select the closest based on threshold
            all_distances.sort()
            closest = all_distances[:threshold]

            # Find the most frequent group ID among the closest particles
            ids = [group_id for _, group_id in closest]
            most_common_id = max(set(ids), key=ids.count)
            result[point] = most_common_id

            # Convert dictionary to list 
            groups_of_interest = list(result.values())
        return groups_of_interest
    
    def calculate_barycenters(self, target_ids=None):
        """
        Calculates barycenters of halos based on particles' positions and corresponding halo group ID.
        Assumes that mass is uniform across all particles.

        Parameters:
            target_ids (optional): List of target group IDs to consider for barycenter computation.
        
        Returns: 
            dict: A dictionary of Group IDs and their corresponding barycenter coordinates. 
        """
        self.load_group_ids()
        self.load_positions()

        # Filter out the group ID for non-member particles
        valid_mask = self.group_ids != 4294967295
        pos = self.positions[valid_mask]
        groupID = self.group_ids[valid_mask]

        if target_ids is not None:
            target_mask = np.isin(groupID, target_ids)
            pos = pos[target_mask]
            groupID = groupID[target_mask]

        # Calculate barycenters
        barycenters = {}
        for uid in np.unique(groupID):
            mask = groupID == uid
            barycenter = np.mean(pos[mask], axis=0)
            barycenters[uid] = barycenter

        return barycenters
    
    def compute_distances_to_barycenters(self, target_id, N=10):
        """
        Computes the distance from the barycenter of the specified group to the barycenters of all other groups,
        and returns the N closest groups.

        Parameters: 
            target_id (int): The target group ID to compute distances from.
            N (int): Number of closest group IDs to return.

        Returns:
            dict: A dictionary of Group IDs as keys and distances to target group as values, in descending distance order.
        """
        self.load_group_ids()
        self.load_positions()
        
        barycenters = self.calculate_barycenters()
        if target_id not in barycenters:
            raise ValueError("Target group ID not found in the data.")

        target_barycenter = barycenters[target_id]
        distances = {}

        for uid, barycenter in barycenters.items():
            if uid != target_id:
                dist = np.linalg.norm(target_barycenter - barycenter)
                distances[uid] = dist

        # Now to get the N closest groups, we sort distances by value and return the top N
        closest_groups = dict(sorted(distances.items(), key=lambda item: item[1])[:N])

        return closest_groups

    def plot_2D_barycenters(self,target_ids=None):
        """
        Plots barycenters on a scatter plot.
        """
        if target_ids is not None:
            barycenter = self.calculate_barycenters(target_ids)

        else:
            barycenter = self.calculate_barycenters()
        # Extract x and y coordinates from each barycenter
        x = [barycenter[0] for barycenter in barycenter.values()]
        y = [barycenter[1] for barycenter in barycenter.values()]

        # Create scatter plot
        plt.figure(figsize=(10,8))
        plt.scatter(x,y, color='blue', marker='o', s=10, alpha=0.5)
        plt.title('Plot of DM halo barycenters')
        plt.xlabel('x (Mpc/h)')
        plt.ylabel('y (Mpc/h)')
        plt.grid(True)

        plt.show()

    def plot_3D_barycenters(self):
        """
        Plots barycenters on a 3D scatter plot.
        """
        barycenter = self.calculate_barycenters()
        # Extract x and y coordinates from each barycenter
        x = [barycenter[0] for barycenter in barycenter.values()]
        y = [barycenter[1] for barycenter in barycenter.values()]
        z = [barycenter[2] for barycenter in barycenter.values()]

        # Create scatter plot
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection='3d')
        scatter = ax.scatter(x,y, z, color='blue', marker='o', s=10, alpha=0.5)
        
        ax.set_title('Plot of DM halo barycenters')
        ax.set_xlabel('x (Mpc/h)')
        ax.set_label('y (Mpc/h)')
        ax.set_zlabel('z (Mpc/h)')
        ax.grid(True)
        plt.show()

    def calculate_angular_momentum(self,target_ids=None):
        """
        Calculates angular momentum of a group of particles.

        Parameters:
            target_ids (int): ID of group for which to calculate angular momentum.

        Returns:
            np.ndarray: Angular momentum vector of the group.
        """
        self.load_positions()
        self.load_velocities()
        self.load_masses()
        self.load_group_ids()

        # Filter particles by group ID
        valid_mask = self.group_ids != 000000000 #4294967295
        pos = (self.positions*3.085678e24)[valid_mask] ## <-------------- weird unit stuff that I'm checking
        ###### pos = self.positions[valid_mask]
        vel = self.velocities[valid_mask]
        mass = self.masses[valid_mask]
        group_id = self.group_ids[valid_mask]

        angular_momentum = {}

        # Compute angular momentum for all or specified groups
        for uid in np.unique(group_id):
            if target_ids is None or uid in target_ids:
                mask = group_id == uid
                r = pos[mask] - np.mean(pos[mask], axis=0)
                v = vel[mask]
                m = mass[mask][:, np.newaxis]  # ensures mass is correctly shaped for multiplication
                L = np.sum( np.cross(r,m*v), axis=0 )
                angular_momentum[uid] = L
                # Instead of storing the 3D coordinates of the angular momenta, store only the magnitude
                #angular_momentum[uid] = np.linalg.norm(L)  
        return angular_momentum
    
    def calculate_system_angular_momentum(self):
        """
        Calculates angular momentum of entire system relative to the center of mass of system.

        Returns:
            ndarray: Total angular momentum vector of the system.
        """
        self.load_velocities()
        self.load_positions()
        self.load_masses()

        # Calculates center of mass for positons and velocity      
        pos_com = np.mean(self.positions)
        vel_com = np.mean(self.positions)
        
        # Calculates relative positons and relative velocities
        r = self.positions - pos_com 
        v = self.velocities - vel_com 

        # Momentum 
        p = self.masses[:, np.newaxis]*v
    
        # Calculate angular momentum
        L = np.cross(r, p)
        return L

    def OLD_calculate_system_angular_momentum(self):
        """
        Calculates angular momentum of entire system relative to the center of mass of system.

        Returns:
            ndarray: Total angular momentum vector of the system.
        """
        self.load_velocities()
        self.load_positions()
        self.load_masses()
        
        relative_pos = self.positions - self.calculate_com()
        
        masses = self.masses[:, np.newaxis]
        
        L_individual = np.cross(relative_pos, masses*self.velocities)
        
        L_total = np.sum(L_individual, axis=0)
        
        return L_total
    
    def calculate_com(self):
        """
        Calculates center of mass for a system of particles.

        Returns:
            ndarray: The center of mass coordinates, shape (3, ).
        """
        self.load_masses()
        self.load_positions()

        masses = self.masses[:,np.newaxis]
        total_mass = np.sum(masses)
        
        weighted_pos_sum = np.sum(masses * self.positions, axis=0)
        
        center_of_mass = weighted_pos_sum / total_mass
        
        return center_of_mass
    
    def plot_angular_momenta(self, angular_momenta_dict):
        """
        Plots 3D vectors of angular momenta for specified groups.

        Parameters:
            angular_momenta_dict (dict): Dictionary where keys are group labels and values are angular momentum vectors.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        origin = [0, 0, 0]

        # Determine the limits for the plot based on the provided vectors
        all_values = np.array(list(angular_momenta_dict.values()))
        x_min, x_max = np.min(all_values[:, 0]), np.max(all_values[:, 0])
        y_min, y_max = np.min(all_values[:, 1]), np.max(all_values[:, 1])
        z_min, z_max = np.min(all_values[:, 2]), np.max(all_values[:, 2])

        colors = plt.cm.jet(np.linspace(0,1, len(angular_momenta_dict)))

        # Plot each vector with its specified label and color
        for (label, vector), color in zip(angular_momenta_dict.items(), colors):
            ax.quiver(*origin, *vector, color=color, label=f'Group {label}')

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.legend()
        plt.show()
    
    def calculate_rms(self, type=0):
        """ 
        Calculates root mean square of provided array of values. 

        Parameters:
            type (int: 0 or 1): DM particle type to calculate RMS of. Select 0 for velocity, 1 for positions/radius
        
        Returns: 
            float: The RMS value.
        """
        if type == 0:
            self.load_velocities()
            # Calculate velocity magnitudes
            vel_mag = np.linalg.norm(self.velocities, axis=1)
            rms = np.sqrt( np.mean( np.square(vel_mag) ) )

        elif type == 1:
            self.load_positions()
            # Calculate relative positons
            relative_pos = self.positions - self.calculate_com()
            # Calculate radii from center of mass
            radii = np.linalg.norm(relative_pos, axis=1)
            rms = np.sqrt( np.mean( np.square(radii) ) )
        return rms
    
    def plot_group_velocities(self, target_group_ids, scale=5000):
        """
        Plots velocities of particles in specified groups in a 2D scatter plot, using class-loaded data.

        Parameters:
            target_group_ids (list of int): Specific group IDs to plot.
        """
        # Load necessary data from class
        self.load_positions()
        self.load_velocities()
        self.load_group_ids()

        # Ensure the target_group_ids is provided and filter data accordingly
        mask = np.isin(self.group_ids, target_group_ids)
        t_pos = self.positions[mask]
        t_vel = self.velocities[mask] / 1e5  # Convert velocities from cm/s to km/s 
        t_group_ids = self.group_ids[mask]

        # Unique group IDs and colormap
        unique_groups = np.unique(t_group_ids)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_groups)))  # Using 'viridis' colormap

        # Create a color dictionary for each group ID
        color_dict = {uid: color for uid, color in zip(unique_groups, colors)}

        # Setup plot
        fig, ax = plt.subplots(figsize=(10,6))

        # Plotting velocities with different colors for different groups
        for group in unique_groups:
            group_mask = t_group_ids == group
            y, z = t_pos[group_mask, 1], t_pos[group_mask, 2]
            v, w = t_vel[group_mask, 1], t_vel[group_mask, 2]

            ax.quiver(y, z, v, w, color=color_dict[group], scale=scale, label=f'Group {group}')

        ax.set_xlabel('Y (Mpc/h)')
        ax.set_ylabel('Z (Mpc/h)')
        ax.legend(title='Group ID')
        plt.grid(True)
        plt.show()

    def get_particle_ids_by_group(self, group_id):
        """
        Returns the individual particle IDs for the specified group ID.

        Parameters:
            group_id (int): The group ID for which we want all particle IDs.

        Returns: 
            ndarray: Array of particle IDs that belong to specified group. 
        """
        self.load_group_ids()
        self.load_particle_ids()

        return self.particle_ids[self.group_ids == group_id]
    
    def plot_particles_by_ids(self, particle_ids_list, labels=None):
        """
        Plots the positon of particiles specified by their IDs.

        Parameters:
            particle_ids_list (list or array-like): List containing arrays of particle IDs to plot.
            labels (list of str, optional): Labels for each particle ID array for the legend.
        """
        self.load_positions()
        self.load_particle_ids()

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.Spectral(np.linspace(0,1, len(particle_ids_list)))

        for idx, particle_ids in enumerate(particle_ids_list):
            mask = np.isin(self.particle_ids, particle_ids)
            x,y,z = self.positions[mask].T
            ax.scatter(x,y,z, s=5, alpha=0.5, color=colors[idx], label=labels[idx] if labels else f'Group {idx}')

        ax.set_xlabel('X (Mpc/h)')
        ax.set_ylabel('Y (Mpc/h)')
        ax.set_zlabel('Z (Mpc/h)')
        if labels:
            ax.legend(title='Particle Groups')
        plt.show()

    def analyze_group_energies(self, group_ids):
        """
        Calculates and compares potential and kinetic energy for specified groups to determine their dynamic state.

        Parameters:
            group_ids (list of int): List of group IDs for which to analyze energy.

        Returns:
            dict: Dictionary containing energy analysis for each group.
        """
        self.load_positions()
        self.load_velocities()
        self.load_masses()
        self.load_potentials()
        self.load_group_ids()

        energy_analysis = {}

        for group_id in group_ids:
            mask = self.group_ids == group_id
            if np.any(mask):
                velocities = self.velocities[mask]
                masses = self.masses[mask]
                potentials = self.potentials[mask]

                # Kinetic energy calculation: KE = 1/2 * m * v^2
                v_squared = np.sum(velocities**2, axis=1)
                kinetic_energy = 0.5 * np.sum(masses * v_squared)

                # Potential energy calculation: PE = sum of potential energy values
                potential_energy = np.sum(potentials * masses)

                # Analyze the energy ratio to determine the system state
                if kinetic_energy * 2 < np.abs(potential_energy):
                    state = 'Bound (Possibly collapsing)'
                elif kinetic_energy * 2 > np.abs(potential_energy):
                    state = 'Unbound (Possibly dispersing)'
                else:
                    state = 'In virial equilibrium'

                energy_analysis[group_id] = {
                    'Kinetic Energy': kinetic_energy,
                    'Potential Energy': potential_energy,
                    'Total Energy': kinetic_energy + potential_energy,
                    'State': state
                }
            else:
                energy_analysis[group_id] = 'Group ID not found or contains no particles'

        return energy_analysis

