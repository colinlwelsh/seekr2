"""
mmvt_drmsd_path_cv.py

Structures, methods, and functions for handling Inter-Group Distance-RMSD (dRMSD) Path CVs.
Supports milestoning along path progress (s) and orthogonal distance (z) individually or
together via Path_combo.
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
    cv_type : str
        's' for path progress (1 to N) or 'z' for orthogonal distance off the path.
    """

    def __init__(self, index, group1, group2, ref_file, lambda_param, cv_type="s"):
        self.index = index
        self.group1 = group1
        self.group2 = group2
        self.ref_file = ref_file
        self.lambda_param = float(lambda_param)
        self.cv_type = cv_type.lower()
        assert self.cv_type in ["s", "z"], \
            f"Invalid cv_type '{cv_type}' for MMVT_dRMSD_Path_CV. Must be 's' or 'z'."
        self.lambda_var = f"lam_{self.index}"

        self.name = f"mmvt_drmsd_path_{self.cv_type}"
        self.openmm_expression = None
        self.restraining_expression = None
        self.cv_expression = None
        self._path_s_expression = None
        self._path_s_definitions = None
        self._path_z_expression = None
        self._path_z_definitions = None
        self.num_groups = 1
        self.per_dof_variables = ["k", "value"]
        self.global_variables = [self.lambda_var]
        self._mygroup_list = None
        self.variable_name = self.cv_type
        self._ref_distances = None
        return

    @classmethod
    def update_blacklist(cls, attr_name):
        """Dynamically update the class blacklist to include a new attribute."""
        blacklist = list(getattr(cls, "_Serializer__blacklist", ()))
        if attr_name not in blacklist:
            blacklist.append(attr_name)
            cls._Serializer__blacklist = tuple(blacklist)
        return

    def __name__(self):
        return "MMVT_dRMSD_Path_CV"

    def _get_cv_type(self):
        return getattr(self, "cv_type", "s").lower()

    def _get_ref_dists(self):
        """Internal helper to access cached reference distances without re-reading PDB files."""
        if not hasattr(self, "_ref_distances") or self._ref_distances is None:
            self._ref_distances = self.get_reference_distances()
            self.update_blacklist("_ref_distances")
        return self._ref_distances

    def _get_path_s_expression(self):
        if not hasattr(self, "_path_s_expression") or self._path_s_expression is None:
            self._path_s_expression, self._path_s_definitions = self.make_path_s_expression()
            self.update_blacklist("_path_s_expression")
            self.update_blacklist("_path_s_definitions")
        return self._path_s_expression, self._path_s_definitions

    def _get_path_z_expression(self):
        if not hasattr(self, "_path_z_expression") or self._path_z_expression is None:
            self._path_z_expression, self._path_z_definitions = self.make_path_z_expression()
            self.update_blacklist("_path_z_expression")
            self.update_blacklist("_path_z_definitions")
        return self._path_z_expression, self._path_z_definitions

    def _get_active_path_expression(self):
        """Returns (expression_str, definition_str) for the active cv_type ('s' or 'z')."""
        if self._get_cv_type() == "z":
            return self._get_path_z_expression()
        return self._get_path_s_expression()

    def _build_exp_definitions(self):
        ref_distances = self._get_ref_dists()
        num_frames = ref_distances.shape[0]
        num_pairs = len(self.group1) * len(self.group2)

        pair_names = []
        for i in range(len(self.group1)):
            p1_name = f"p{i+1}"
            for j in range(len(self.group2)):
                p2_name = f"p{len(self.group1)+j+1}"
                name = f"distance({p1_name}, {p2_name})"
                pair_names.append((name, i, j))

        exp_terms = []
        sumsq = " + ".join(f"{name}^2" for name, i, j in pair_names)
        for k in range(num_frames):
            cross = " + ".join(f"{name}*{ref_distances[k,i,j]:.6f}" for name, i, j in pair_names)
            const_k = float(np.sum(ref_distances[k] ** 2)) / num_pairs
            dmsd_term = f"({sumsq} - 2*({cross})) / {num_pairs} + {const_k:.6f}"
            exp_terms.append(f"e_{k} = exp(-{self.lambda_var} * ({dmsd_term}))")

        den_terms = [f"e_{k}" for k in range(num_frames)]
        denominator = " + ".join(den_terms)
        definition_str = "; ".join(exp_terms)
        return num_frames, denominator, definition_str

    def make_path_z_expression(self):
        """Creates OpenMM expression and definitions for orthogonal path distance z(x)."""
        _, denominator, definition_str = self._build_exp_definitions()
        expression_str = f"(-1.0 / {self.lambda_var} * log(max(1e-300, {denominator})))"
        return expression_str, definition_str

    def make_path_s_expression(self, var_tol=0.005):
        """Creates OpenMM expression and definitions for path progress s(x)."""
        num_frames, denominator, definition_str = self._build_exp_definitions()
        num_terms = [f"{k + 1} * e_{k}" for k in range(num_frames)]
        numerator = " + ".join(num_terms)
        expression_str = f"({numerator}) / ({denominator})"
        return expression_str, definition_str

    def make_boundary_force(self, alias_id):
        """Creates an MMVT boundary force for the active path variable (S or Z)."""
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        assert self.num_groups == 1
        assert len(self.group1) > 0 and len(self.group2) > 0, \
            "Both group1 and group2 must contain atoms."

        path_expression, path_definitions = self._get_active_path_expression()
        self.openmm_expression = f"step(k*({path_expression} - value))"
        expression_w_bitcode = f"bitcode*({self.openmm_expression}); {path_definitions}"

        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        boundary_force = openmm.CustomCompoundBondForce(len(all_particles), expression_w_bitcode)
        return boundary_force

    def make_restraining_force(self, alias_id):
        """
        Creates harmonic restraint for the active CV:
        - For cv_type='s': harmonic restraint on S plus optional flat-bottom upper bound on Z.
        - For cv_type='z': harmonic restraint on Z.
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        num_particles = len(all_particles)

        if self._get_cv_type() == "z":
            path_z_expression, definitions = self._get_path_z_expression()
            self.restraining_expression = f"0.5*k*({path_z_expression} - value)^2"
        else:
            path_s_expression, definitions = self._get_path_s_expression()
            self.restraining_expression = (f"0.5*k*({path_s_expression} - value)^2")


        expression_w_bitcode = f"bitcode*({self.restraining_expression}); {definitions}"
        restraining_force = openmm.CustomCompoundBondForce(num_particles, expression_w_bitcode)
        return restraining_force

    def make_cv_force(self, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        path_expression, path_definitions = self._get_active_path_expression()
        expression = f"{path_expression}; {path_definitions}"
        num_particles = len(self.group1) + len(self.group2)
        return openmm.CustomCompoundBondForce(num_particles, expression)

    def make_voronoi_cv_boundary_forces(self, me_val, neighbor_val, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        path_expression, path_definitions = self._get_path_s_expression()
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        num_particles = len(all_particles)

        me_param = f"me_val_{self.index}_alias_{alias_id}"
        me_expr = f"({me_param} - {path_expression})^2; {path_definitions}"
        me_force = openmm.CustomCompoundBondForce(num_particles, me_expr)
        me_force.addGlobalParameter(me_param, me_val)
        me_force.addGlobalParameter(self.lambda_var, self.lambda_param)
        me_force.addBond(all_particles, [])

        neighbor_param = f"neighbor_val_{self.index}_alias_{alias_id}"
        neighbor_expr = f"({neighbor_param} - {path_expression})^2; {path_definitions}"
        neighbor_force = openmm.CustomCompoundBondForce(num_particles, neighbor_expr)
        neighbor_force.addGlobalParameter(neighbor_param, neighbor_val)
        neighbor_force.addGlobalParameter(self.lambda_var, self.lambda_param)
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
        force.addGlobalParameter(self.lambda_var, self.lambda_param)
        force.addPerBondParameter("bitcode")
        force.addPerBondParameter("k")
        force.addPerBondParameter("value")
        return [self.lambda_var]

    def add_groups_and_variables(self, force, variables, alias_id):
        assert force.getNumBonds() == 0, \
            "add_groups_and_variables cannot be called twice for the same CV."
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        force.addBond(all_particles, variables)   # bitcode, k, value as real per-bond values
        return

    def update_groups_and_variables(self, force, variables, alias_id, context):
        all_particles = [int(a) for a in self.group1] + [int(a) for a in self.group2]
        force.setBondParameters(0, all_particles, variables)
        return

    def get_variable_values_list(self, milestone):
        assert milestone.cv_index == self.index
        values_list = []
        bitcode = 2 ** (milestone.alias_index - 1)
        k = milestone.variables["k"] * unit.kilojoules_per_mole / unit.nanometers**2
        value = milestone.variables["value"]
        values_list.extend([bitcode, k, value])
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

    def get_mdtraj_s_z_values(self, traj, frame_index):
        """Returns tuple (s_val, z_val) for a trajectory frame."""
        drmsds = self._get_frame_drmsds_mdtraj(traj, frame_index)
        return self._calc_path_s_z(drmsds)

    def get_mdtraj_cv_value(self, traj, frame_index):
        """Returns scalar CV value (s_val or z_val) matching self.cv_type."""
        s_val, z_val = self.get_mdtraj_s_z_values(traj, frame_index)
        return z_val if self._get_cv_type() == "z" else s_val

    def get_openmm_context_s_z_values(self, context, positions=None, ref_distances=None, verbose=False, system=None, tolerance=0.0):
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

        drmsds = np.array([
            np.mean((current_dists - ref_distances[k]) ** 2)
            for k in range(ref_distances.shape[0])
        ])
        s_val, z_val = self._calc_path_s_z(drmsds)

        assert np.isfinite(s_val) and np.isfinite(z_val), "Non-finite value detected."
        return (s_val, z_val)

    def get_openmm_context_cv_value(self, context, positions=None, ref_distances=None, verbose=False, system=None, tolerance=0.0):
        """Returns scalar CV value (s_val or z_val) evaluated from an OpenMM Context."""
        s_val, z_val = self.get_openmm_context_s_z_values(
            context, positions=positions, ref_distances=ref_distances,
            verbose=verbose, system=system, tolerance=tolerance
        )
        return z_val if self._get_cv_type() == "z" else s_val

    def check_mdtraj_within_boundary(self, traj, milestone_variables, verbose=False, TOL=0.001):
        for frame_index in range(traj.n_frames):
            value = self.get_mdtraj_cv_value(traj, frame_index)
            result = self.check_value_within_boundary(value, milestone_variables, verbose=verbose, tolerance=TOL)
            if not result:
                return False
        return True

    def check_openmm_context_within_boundary(self, context, milestone_variables, positions=None, ref_distances=None, verbose=False, tolerance=0.0):
        value = self.get_openmm_context_cv_value(
            context, positions=positions, ref_distances=ref_distances,
            verbose=verbose, tolerance=tolerance
        )
        return self.check_value_within_boundary(value, milestone_variables, verbose=verbose, tolerance=tolerance)

    def check_value_within_boundary(self, value, milestone_variables, verbose=False, tolerance=0.0):
        if isinstance(value, (tuple, list)):
            cv_val = value[1] if self._get_cv_type() == "z" else value[0]
        else:
            cv_val = float(value)

        milestone_k = milestone_variables["k"]
        milestone_value = milestone_variables["value"]

        if milestone_k * (cv_val - milestone_value) > tolerance:
            if verbose:
                label = self._get_cv_type().upper()
                print(f"dRMSD Path {label} value ({cv_val:.4f}) exceeded boundary ({milestone_value:.4f}).")
            return False

        return True

    def check_mdtraj_close_to_boundary(self, traj, milestone_variables, verbose=False, max_avg=0.03, max_std=0.05):
        diffs = []
        for frame_index in range(traj.n_frames):
            cv_val = self.get_mdtraj_cv_value(traj, frame_index)
            milestone_value = milestone_variables["value"]
            diffs.append(cv_val - milestone_value)

        avg_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if abs(avg_diff) > max_avg or std_diff > max_std:
            if verbose:
                print(f"Average diff: {avg_diff:.4f}, std: {std_diff:.4f}.")
            return False
        return True

    def get_atom_groups(self):
        return [self.group1, self.group2]

    def get_variable_values(self):
        return []
    
    def get_reference_distances(self):
        """Loads reference PDB trajectory once to precalculate pairwise distance matrices."""
        assert os.path.exists(self.ref_file), \
            "File {} does not exist. Make sure ".format(self.ref_file) \
            +"that any programs using the get_mdtraj_cv_value() method " \
            "within an API is performed in the model directory."
        ref_traj = mdtraj.load(self.ref_file)
        ref_dists = np.zeros(shape=(ref_traj.n_frames, len(self.group1), len(self.group2)))
        for k in range(ref_traj.n_frames):
            coords1_ref = ref_traj.xyz[k, self.group1, :]
            coords2_ref = ref_traj.xyz[k, self.group2, :]
            diff_ref = coords1_ref[:, np.newaxis, :] - coords2_ref[np.newaxis, :, :]
            ref_dists[k] = np.sqrt(np.sum(diff_ref ** 2, axis=-1))
        return ref_dists

    def get_distances_at_value(self, value):
        value_idx = int(np.floor(value)-1)
        dists = self._get_ref_dists()
        return dists[value_idx]


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

    cv_type = getattr(drmsd_path_cv_input, "cv_type", "s")
    cv = MMVT_dRMSD_Path_CV(
        index=index,
        group1=group1,
        group2=group2,
        ref_file=ref_file_basename,
        lambda_param=drmsd_path_cv_input.lambda_param,
        cv_type=cv_type,
    )
    if getattr(drmsd_path_cv_input, "variable_name", None):
        cv.variable_name = drmsd_path_cv_input.variable_name
    return cv
