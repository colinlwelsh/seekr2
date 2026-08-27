"""
Structures, methods, and functions for handling Inter-Group Distance-RMSD (dRMSD) Path CVs.
Includes both path progress (s) tracking and orthogonal distance (z) upper-bounding.
"""

import os
import numpy as np
import mdtraj

from parmed import unit

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base
from seekr2.modules.mmvt_cvs.mmvt_cv_base import MMVT_collective_variable


class MMVT_dRMSD_Path_CV(MMVT_collective_variable):
    """
    An inter-group Distance-RMSD (dRMSD) Path collective variable tracking progress 
    s(x) along a path, with an enforced upper bound on orthogonal distance z(x).

    Attributes
    ----------
    index : int
        Index of this collective variable within SEEKR2.
    group1 : list of int
        Atom indices for the first group (e.g., ligand atoms).
    group2 : list of int
        Atom indices for the second group (e.g., protein binding pocket atoms).
    ref_file : str
        Path to multi-frame PDB/trajectory file containing reference frames.
    lambda_param : float
        Exponential smoothing factor (in nm^-2).
    """

    def __init__(self, index, group1, group2, ref_file, lambda_param):
        self.index = index
        self.group1 = group1
        self.group2 = group2
        self.ref_file = ref_file
        self.lambda_param = float(lambda_param)
            
        self.name = "mmvt_drmsd_path"
        self.openmm_expression = None
        self.restraining_expression = None
        self.cv_expression = "PATH_S"
        self._path_s_expression = None
        self._path_s_definitions = None
        self._path_z_expression = None
        self.num_groups = 1
        self.per_dof_variables = ["k", "value", "k_z", "z_cutoff"]
        self.global_variables = ["lam"]
        self._mygroup_list = None
        self.variable_name = "v"
        self._ref_distances = None
        return

    @classmethod
    def update_blacklist(cls, attr_name):
        """Dynamically update the class blacklist to include a new attribute."""
        blacklist = list(getattr(cls, "_Serializer__blacklist", ()))
        blacklist.append(attr_name)
        cls._Serializer__blacklist = tuple(blacklist)
        return

    def __name__(self):
        return "MMVT_dRMSD_Path_CV"

    def _get_ref_dists(self):
        """Internal helper to access cached reference distances without re-reading PDB files."""
        if not hasattr(self, '_ref_distances') or self._ref_distances is None:
            self._ref_distances = self.get_reference_distances()
            self.update_blacklist("_ref_distances")
        return self._ref_distances

    def _get_path_s_expression(self):
        if not hasattr(self, '_path_s_expression') or self._path_s_expression is None:
            self._path_s_expression, self._path_s_definitions = self.make_path_s_expression()
            self.update_blacklist('_path_s_expression')
        return self._path_s_expression, self._path_s_definitions

    def _get_path_z_expression(self):
        if not hasattr(self, '_path_z_expression') or self._path_z_expression is None:
            self._path_z_expression = self.make_path_z_expression()
            self.update_blacklist('_path_z_expression')
        return self._path_z_expression

    def make_path_z_expression(self):
        """Creates an OpenMM CustomCVForce representing path progress s(x)."""
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        num_frames = self._get_ref_dists().shape[0]

        den_terms = [f"e_{k}" for k in range(num_frames)]
        denominator = " + ".join(den_terms)
        expression = f"(-1.0 / lam * log({denominator}))"
        return expression

    def make_path_s_expression(self):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        ref_distances = self._get_ref_dists()
        num_frames = ref_distances.shape[0]
        num_pairs = len(self.group1) * len(self.group2)

        distance_terms = []
        pair_names = []
        for i in range(len(self.group1)):
            p1_name = f"p{i+1}"
            for j in range(len(self.group2)):
                p2_name = f"p{len(self.group1)+j+1}"
                name = f"distance({p1_name}, {p2_name})"
                pair_names.append((name,i,j))

        exp_terms = []
        sumsq   = " + ".join(f"{name}^2" for name,i,j in pair_names)
        for k in range(num_frames):
            cross = " + ".join(f"{name}*{ref_distances[k,i,j]:.6f}" for name,i,j in pair_names)
            const_k = float(np.sum(ref_distances[k]**2)) / num_pairs
            dmsd_term = f"({sumsq} - 2*({cross})) / {num_pairs} + {const_k:.6f}"
            exp_terms.append(f"e_{k} = exp(-lam * ({dmsd_term}))")

        num_terms = [f"{k + 1} * e_{k}" for k in range(num_frames)]
        den_terms = [f"e_{k}" for k in range(num_frames)]
        numerator = " + ".join(num_terms)
        denominator = " + ".join(den_terms)
        definition_str = "; ".join(exp_terms)
        expression_str = f"({numerator}) / ({denominator})"
        return expression_str, definition_str

    def make_boundary_force(self, alias_id):
        """
        Creates a boundary force enforcing S milestone progress boundary 
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        assert self.num_groups == 1
        
        ref_distances = self._get_ref_dists()
        num_frames = ref_distances.shape[0]
        num_atoms1 = len(self.group1)
        num_atoms2 = len(self.group2)

        assert num_atoms1 > 0 and num_atoms2 > 0, "Both group1 and group2 must contain atoms."
        num_pairs = num_atoms1 * num_atoms2

        path_s_expression, path_s_definitions = self._get_path_s_expression()

        self.openmm_expression = (
            f"step(k*({path_s_expression} - value))"
        )
        expression_w_bitcode = f"bitcode*({self.openmm_expression}); {path_s_definitions}"
        #print(expression_w_bitcode)
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        num_particles = len(all_particles)

        boundary_force = openmm.CustomCompoundBondForce(num_particles, expression_w_bitcode)

        return boundary_force

    def make_restraining_force(self, alias_id):
        """Creates harmonic restraint for S and flat-bottom upper-bound restraint for Z."""
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        path_s_expression, definitions = self._get_path_s_expression()
        path_z_expression = self._get_path_z_expression()
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        num_particles = len(all_particles)

        self.restraining_expression = (
            f"0.5*k*({path_s_expression} - value)^2 + "
            f"0.5*k_z*(max(0, {path_z_expression} - z_cutoff))^2"
        )
        expression_w_bitcode = f"bitcode*({self.restraining_expression}); {definitions}"

        restraining_force = openmm.CustomCompoundBondForce(num_particles, expression_w_bitcode)
        return restraining_force

    def make_cv_force(self, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        path_s_expression, path_s_definitions = self._get_path_s_expression()

        expression = f"{path_s_expression}; {path_s_definitions}"
        num_particles = len(self.group1) + len(self.group2)

        return openmm.CustomCompoundBondForce(num_particles, expression)

    def make_voronoi_cv_boundary_forces(self, me_val, neighbor_val, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        path_s_expression, path_s_definitions = self._get_path_s_expression()
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        num_particles = len(all_particles)

        me_param = f"me_val_{self.index}_alias_{alias_id}"
        me_expr = f"({me_param} - {path_s_expression})^2; {path_s_definitions}"
        me_force = openmm.CustomCompoundBondForce(num_particles, me_expr)
        me_force.addGlobalParameter(me_param, me_val)
        me_force.addGlobalParameter("lam", self.lambda_param)
        me_force.addBond(all_particles, [])

        neighbor_param = f"neighbor_val_{self.index}_alias_{alias_id}"
        neighbor_expr = f"({neighbor_param} - {path_s_expression})^2; {path_s_definitions}"
        neighbor_force = openmm.CustomCompoundBondForce(num_particles, neighbor_expr)
        neighbor_force.addGlobalParameter(neighbor_param, neighbor_val)
        neighbor_force.addGlobalParameter("lam", self.lambda_param)
        neighbor_force.addBond(all_particles, [])

        return me_force, neighbor_force

    def update_voronoi_cv_boundary_forces(self, me_force, me_val, neighbor_force, neighbor_val, alias_id, context):
        context.setParameter(f"me_val_{self.index}_alias_{alias_id}", me_val)
        context.setParameter(f"neighbor_val_{self.index}_alias_{alias_id}", neighbor_val)
        return

    def make_namd_colvar_string(self):
        raise Exception("MMVT Path CVs are not available in NAMD")

    def add_groups(self, force):
        return

    def add_parameters(self, force):
        force.addGlobalParameter("lam", self.lambda_param)
        force.addPerBondParameter("bitcode")
        force.addPerBondParameter("k")
        force.addPerBondParameter("value")
        force.addPerBondParameter("k_z")
        force.addPerBondParameter("z_cutoff")
        return ["lam"]

    def add_groups_and_variables(self, force, variables, alias_id):
        assert force.getNumBonds() == 0, \
            "add_groups_and_variables cannot be called twice for the same CV."
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        force.addBond(all_particles, variables)   # bitcode, k, value, k_z, z_cutoff as real per-bond values
        return

    def update_groups_and_variables(self, force, variables, alias_id, context):
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        force.setBondParameters(0, all_particles, variables)
        # NOTE: caller must call context.reinitialize() (or force.updateParametersInContext(context)
        # if only per-bond *parameters*, not particle count, changed) after this
        return

    def get_variable_values_list(self, milestone):
        assert milestone.cv_index == self.index
        values_list = []
        bitcode = 2**(milestone.alias_index - 1)
        k = milestone.variables["k"] * unit.kilojoules_per_mole / unit.nanometers**2
        value = milestone.variables["value"]
        
        k_z_val = milestone.variables.get("k_z", milestone.variables.get("k", 0.0))
        k_z = k_z_val * unit.kilojoules_per_mole / unit.nanometers**2
        
        z_cutoff = milestone.variables.get("z_cutoff", milestone.variables.get("z_value", 999.0))
        
        values_list.extend([bitcode, k, value, k_z, z_cutoff])
        return values_list

    def get_namd_evaluation_string(self, milestone, cv_val_var="cv_val"):
        raise Exception("MMVT Path CVs are not available in NAMD")

    def _get_frame_drmsds_mdtraj(self, traj, frame_index):
        """Calculates inter-group dRMSD to each reference frame via MDTraj."""
        ref_distances = self._get_ref_dists()
        num_frames = ref_distances.shape[0]

        coords1_curr = traj.xyz[frame_index, self.group1, :]
        coords2_curr = traj.xyz[frame_index, self.group2, :]
        diff_curr = coords1_curr[:, np.newaxis, :] - coords2_curr[np.newaxis, :, :]
        current_dists = np.sqrt(np.sum(diff_curr ** 2, axis=-1))

        drmsds = np.zeros(num_frames)
        for k in range(num_frames):
            ref_dists = ref_distances[k]
            drmsds[k] = np.mean((current_dists - ref_dists) ** 2)

        return drmsds

    def _calc_path_s_z(self, drmsds):
        exps = np.exp(-self.lambda_param * (drmsds))
        sum_exps = np.sum(exps)

        weights = np.arange(1, len(drmsds) + 1)
        if sum_exps == 0:
            sum_exps = np.finfo(np.float64).tiny
        s_val = float(np.sum(weights * exps) / sum_exps)
        z_val = float(-1.0 / self.lambda_param * np.log(sum_exps))
        return (s_val, z_val)

    def get_mdtraj_cv_value(self, traj, frame_index):
        """Returns tuple (s_val, z_val) for a trajectory frame."""
        drmsds = self._get_frame_drmsds_mdtraj(traj, frame_index)
        s_val, z_val = self._calc_path_s_z(drmsds)
        return (s_val, z_val)

    def get_openmm_context_cv_value(self, context, positions=None, ref_distances=None, verbose=False, system=None, tolerance=0.0):
        """Returns tuple (s_val, z_val) evaluated from an OpenMM Context."""
        if system is None:
            system = context.getSystem()
        if positions is None:
            state = context.getState(getPositions=True)
            positions = state.getPositions()

        if ref_distances is None:
            ref_distances = self._get_ref_dists()

        pos1 = np.array([positions[i].value_in_unit(unit.nanometers) for i in self.group1])
        pos2 = np.array([positions[i].value_in_unit(unit.nanometers) for i in self.group2])
        
        diff_curr = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        current_dists = np.sqrt(np.sum(diff_curr ** 2, axis=-1))

        drmsds = []
        for k in range(ref_distances.shape[0]):
            ref_dists = ref_distances[k]
            #drmsd_k = np.sqrt(np.mean((current_dists - ref_dists) ** 2))
            drmsd_k = np.mean((current_dists - ref_dists) ** 2)
            drmsds.append(drmsd_k)

        drmsds = np.array(drmsds)
        s_val, z_val = self._calc_path_s_z(drmsds)

        assert np.isfinite(s_val) and np.isfinite(z_val), "Non-finite value detected."
        return (s_val, z_val)

    def check_mdtraj_within_boundary(self, traj, milestone_variables, verbose=False, TOL=0.001):
        for frame_index in range(traj.n_frames):
            value = self.get_mdtraj_cv_value(traj, frame_index)
            result = self.check_value_within_boundary(value, milestone_variables, verbose=verbose, tolerance=TOL)
            if not result:
                return False
        return True

    def check_openmm_context_within_boundary(self, context, milestone_variables, positions=None, ref_distances=None, verbose=False, tolerance=0.0):
        value = self.get_openmm_context_cv_value(context, positions=positions, ref_distances=ref_distances, verbose=verbose, tolerance=tolerance)
        #print(value)
        return self.check_value_within_boundary(value, milestone_variables, verbose=verbose, tolerance=tolerance)

    def check_value_within_boundary(self, value, milestone_variables, verbose=False, tolerance=0.0):
        if isinstance(value, (tuple, list)):
            s_val, z_val = value[0], value[1]
        else:
            s_val, z_val = value, 0.0

        milestone_k = milestone_variables["k"]
        milestone_value = milestone_variables["value"]

        # Check S boundary
        if milestone_k * (s_val - milestone_value) > tolerance:
            if verbose:
                print(f"dRMSD Path S value ({s_val:.4f}) exceeded boundary ({milestone_value:.4f}).")
            return False

        # Check Z upper bound if present in milestone definition
        z_cutoff = milestone_variables.get("z_cutoff", milestone_variables.get("z_value", None))
        if z_cutoff is not None:
            if z_val - z_cutoff > tolerance:
                if verbose:
                    print(f"dRMSD Path Z value ({z_val:.4f}) exceeded cutoff ({z_cutoff:.4f}).")
                return False

        #print('check value function', value)
        return True

    def check_mdtraj_close_to_boundary(self, traj, milestone_variables, verbose=False, max_avg=0.03, max_std=0.05):
        diffs = []
        for frame_index in range(traj.n_frames):
            s_val, _ = self.get_mdtraj_cv_value(traj, frame_index)
            milestone_value = milestone_variables["value"]
            diffs.append(s_val - milestone_value)

        avg_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if abs(avg_diff) > max_avg or std_diff > max_std:
            if verbose:
                print(f"Average diff: {avg_diff:.4f} nm, std: {std_diff:.4f} nm.")
            return False
        return True

    def get_atom_groups(self):
        return [self.group1, self.group2]

    def get_variable_values(self):
        return []
    
    def get_reference_distances(self):
        """Loads reference PDB trajectory once to precalculate pairwise distance matrices."""
        assert os.path.exists(self.ref_file), f"File {self.ref_file} does not exist."
        ref_traj = mdtraj.load(self.ref_file)
        ref_dists = np.zeros(shape=(ref_traj.n_frames, len(self.group1), len(self.group2)))
        for k in range(ref_traj.n_frames):
            coords1_ref = ref_traj.xyz[k, self.group1, :]
            coords2_ref = ref_traj.xyz[k, self.group2, :]
            diff_ref = coords1_ref[:, np.newaxis, :] - coords2_ref[np.newaxis, :, :]
            ref_dists[k] = np.sqrt(np.sum(diff_ref ** 2, axis=-1))
        return ref_dists

def make_mmvt_drmsd_path_cv_object(drmsd_path_cv_input, index, root_directory):
    """
    Helper function to parse XML input and create an inter-group MMVT_dRMSD_Path_CV object.
    """
    import shutil
    ref_file_basename = f"drmsd_path_reference_cv_{index}.pdb"
    group1 = base.parse_xml_list(drmsd_path_cv_input.group1)
    group2 = base.parse_xml_list(drmsd_path_cv_input.group2)
    
    absolute_ref_file = os.path.join(root_directory, ref_file_basename)
    shutil.copyfile(drmsd_path_cv_input.ref_file, absolute_ref_file)
    
    cv = MMVT_dRMSD_Path_CV(
        index=index,
        group1=group1,
        group2=group2,
        ref_file=ref_file_basename,
        lambda_param=drmsd_path_cv_input.lambda_param
    )
    return cv
