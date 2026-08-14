"""
Structures, methods, and functions for handling Path RMSD CVs (s and z).
"""

import os
import numpy as np
from scipy.spatial import transform
import mdtraj

from parmed import unit

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base
from seekr2.modules.mmvt_cvs.mmvt_cv_base import MMVT_collective_variable


class MMVT_Path_CV(MMVT_collective_variable):
    """
    A Path collective variable tracking progress s(x) or orthogonal distance z(x)
    relative to a set of reference structures defining a path.

    Attributes
    ----------
    index : int
        Index of this collective variable within SEEKR2.
    group : list of int
        Atom indices used to calculate the RMSD.
    ref_file : str
        Path to multi-frame PDB/trajectory file containing reference frames.
    lambda_param : float
        Exponential smoothing factor (in nm^-2, typically 2.3 / d^2).
    cv_type : str
        's' for path progress (1 to N) or 'z' for distance off the path.
    align_group : list of int or None
        Atom indices used to align structures before calculating RMSD.
        If None, `group` is used for alignment.
    """

    def __init__(self, index, group, ref_file, lambda_param, cv_type="s", align_group=None):
        self.index = index
        self.group = group
        self.align_group = align_group
        self.ref_file = ref_file
        self.lambda_param = float(lambda_param)
        self.cv_type = cv_type.lower()
        
        if self.cv_type not in ["s", "z"]:
            raise ValueError("cv_type must be either 's' or 'z'.")
            
        self.name = f"mmvt_path_{self.cv_type}"
        self.openmm_expression = None
        self.restraining_expression = None
        self.cv_expression = "PATH"
        self.num_groups = 1
        self.per_dof_variables = []
        self.global_variables = ["k", "value"]
        self._mygroup_list = None
        self.variable_name = "v"
        self._ref_traj = None
        return

    @classmethod
    def update_blacklist(cls, attr_name):
        """Dynamically update the class blacklist to include a new attribute."""
        blacklist = list(getattr(cls, "_Serializer__blacklist", ()))
        blacklist.append(attr_name)
        cls._Serializer__blacklist = tuple(blacklist)
        return

    def __name__(self):
        return f"MMVT_Path_{self.cv_type.upper()}_CV"

    def make_path_cv_force(self):
        """
        Creates the CustomCVForce representing s(x) or z(x) across all reference frames.
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        try:
            import openmm.app as openmm_app
        except ImportError:
            import simtk.openmm.app as openmm_app

        ref_traj = mdtraj.load(self.ref_file)
        num_frames = ref_traj.n_frames

        # Construct exponential terms for each frame: exp(-lam * rmsd_i^2)
        exp_terms = [f"exp(-lam * rmsd_{i}^2)" for i in range(num_frames)]

        if self.cv_type == "s":
            num_terms = [f"{i + 1} * {term}" for i, term in enumerate(exp_terms)]
            numerator = " + ".join(num_terms)
            denominator = " + ".join(exp_terms)
            expression = f"({numerator}) / ({denominator})"
        elif self.cv_type == "z":
            sum_exp = " + ".join(exp_terms)
            expression = f"-1.0 / lam * log({sum_exp})"

        path_cv_force = openmm.CustomCVForce(expression)
        path_cv_force.addGlobalParameter("lam", self.lambda_param)

        # Check alignment plugin requirements
        use_rmsd_plus = self.align_group is not None
        if use_rmsd_plus:
            try:
                import rmsdplusforceplugin
            except ImportError:
                print("Unable to load RMSDPlusForcePlugin. Please install "
                      "from: https://github.com/seekrcentral/rmsdplusforceplugin.git")
                exit()

        for i in range(num_frames):
            frame_pos = ref_traj.xyz[i] * unit.nanometers
            if not use_rmsd_plus:
                rmsd_sub_force = openmm.RMSDForce(frame_pos, self.group)
            else:
                rmsd_sub_force = rmsdplusforceplugin.RMSDPlusForce(
                    frame_pos, self.align_group, self.group
                )
            path_cv_force.addCollectiveVariable(f"rmsd_{i}", rmsd_sub_force)

        return path_cv_force

    def make_boundary_force(self, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        assert self.num_groups == 1
        self.openmm_expression = "step(k_{}*(PATH - value_{}))".format(alias_id, alias_id)
        expression_w_bitcode = "bitcode_{}*".format(alias_id) + self.openmm_expression
        return openmm.CustomCVForce(expression_w_bitcode)

    def make_restraining_force(self, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        assert self.num_groups == 1
        self.restraining_expression = "0.5*k_{}*(PATH - value_{})^2".format(alias_id, alias_id)
        return openmm.CustomCVForce(self.restraining_expression)

    def make_cv_force(self, alias_id):
        return self.make_path_cv_force()

    def make_voronoi_cv_boundary_forces(self, me_val, neighbor_val, alias_id):
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm

        path_me_cv = self.make_path_cv_force()
        path_neighbor_cv = self.make_path_cv_force()

        me_expr = "(me_val_{}_alias_{} - {})^2".format(self.index, alias_id, self.cv_expression)
        me_force = openmm.CustomCVForce(me_expr)
        me_force.addGlobalParameter("me_val_{}_alias_{}".format(self.index, alias_id), me_val)
        me_force.addCollectiveVariable("PATH", path_me_cv)

        neighbor_expr = "(neighbor_val_{}_alias_{} - {})^2".format(self.index, alias_id, self.cv_expression)
        neighbor_force = openmm.CustomCVForce(neighbor_expr)
        neighbor_force.addGlobalParameter("neighbor_val_{}_alias_{}".format(self.index, alias_id), neighbor_val)
        neighbor_force.addCollectiveVariable("PATH", path_neighbor_cv)

        return me_force, neighbor_force

    def update_voronoi_cv_boundary_forces(self, me_force, me_val, neighbor_force, neighbor_val, alias_id, context):
        context.setParameter("me_val_{}_alias_{}".format(self.index, alias_id), me_val)
        context.setParameter("neighbor_val_{}_alias_{}".format(self.index, alias_id), neighbor_val)
        return

    def make_namd_colvar_string(self):
        raise Exception("MMVT Path CVs are not available in NAMD")

    def add_groups(self, force):
        return

    def add_parameters(self, force):
        path_force = self.make_path_cv_force()
        force.addCollectiveVariable("PATH", path_force)
        return

    def add_groups_and_variables(self, force, variables, alias_id):
        if len(variables) >= 3:
            force.addGlobalParameter("bitcode_{}".format(alias_id), variables[0])
            force.addGlobalParameter("k_{}".format(alias_id), variables[1])
            force.addGlobalParameter("value_{}".format(alias_id), variables[2])
        return

    def update_groups_and_variables(self, force, variables, alias_id, context):
        if len(variables) >= 3:
            context.setParameter("bitcode_{}".format(alias_id), variables[0])
            context.setParameter("k_{}".format(alias_id), variables[1])
            context.setParameter("value_{}".format(alias_id), variables[2])
        return

    def get_variable_values_list(self, milestone):
        assert milestone.cv_index == self.index
        values_list = []
        bitcode = 2**(milestone.alias_index - 1)
        k = milestone.variables["k"] * unit.kilojoules_per_mole / unit.nanometers**2
        value = milestone.variables["value"]
        values_list.append(bitcode)
        values_list.append(k)
        values_list.append(value)
        return values_list

    def get_namd_evaluation_string(self, milestone, cv_val_var="cv_val"):
        raise Exception("MMVT Path CVs are not available in NAMD")

    def _get_frame_rmsds_mdtraj(self, traj, frame_index):
        """Helper function to calculate RMSD to each reference frame via MDTraj."""
        align_group = self.align_group if self.align_group is not None else self.group
        
        assert os.path.exists(self.ref_file), f"File {self.ref_file} does not exist."
        ref_traj = mdtraj.load(self.ref_file)
        
        rmsds = np.zeros(ref_traj.n_frames)
        current_frame = traj[frame_index]

        for i in range(ref_traj.n_frames):
            ref_frame = ref_traj[i]
            # Superpose using align_group
            aligned_frame = current_frame.superpose(ref_frame, atom_indices=align_group)
            # Calculate RMSD using group
            rmsd_val = mdtraj.rmsd(aligned_frame, ref_frame, atom_indices=self.group)[0]
            rmsds[i] = rmsd_val

        return rmsds

    def get_mdtraj_cv_value(self, traj, frame_index):
        rmsds = self._get_frame_rmsds_mdtraj(traj, frame_index)
        exps = np.exp(-self.lambda_param * (rmsds**2))
        sum_exps = np.sum(exps)

        if self.cv_type == "s":
            weights = np.arange(1, len(rmsds) + 1)
            return float(np.sum(weights * exps) / sum_exps)
        elif self.cv_type == "z":
            return float(-1.0 / self.lambda_param * np.log(sum_exps))

    def get_openmm_context_cv_value(self, context, positions=None, ref_positions=None, verbose=False, system=None, tolerance=0.0):
        if system is None:
            system = context.getSystem()
        if positions is None:
            state = context.getState(getPositions=True)
            positions = state.getPositions()

        ref_traj = mdtraj.load(self.ref_file)
        align_group = self.align_group if self.align_group is not None else self.group

        # Extract current frame position arrays
        pos_align = np.array([positions[i].value_in_unit(unit.nanometers) for i in align_group])
        pos_rmsd = np.array([positions[i].value_in_unit(unit.nanometers) for i in self.group])

        pos_align_center = np.mean(pos_align, axis=0)
        pos_align_centered = pos_align - pos_align_center
        pos_rmsd_centered = pos_rmsd - pos_align_center

        rmsds = []
        for i in range(ref_traj.n_frames):
            ref_xyz = ref_traj.xyz[i]
            ref_align = ref_xyz[align_group]
            ref_rmsd = ref_xyz[self.group]

            ref_align_center = np.mean(ref_align, axis=0)
            ref_align_centered = ref_align - ref_align_center
            ref_rmsd_centered = ref_rmsd - ref_align_center

            # Calculate optimal rotation based on align_group
            rotation, _ = transform.Rotation.align_vectors(ref_align_centered, pos_align_centered)
            new_pos_rmsd = rotation.apply(pos_rmsd_centered)

            # Compute RMSD over group
            diff = new_pos_rmsd - ref_rmsd_centered
            rmsd_i = np.sqrt(np.sum(diff**2) / len(self.group))
            rmsds.append(rmsd_i)

        rmsds = np.array(rmsds)
        exps = np.exp(-self.lambda_param * (rmsds**2))
        sum_exps = np.sum(exps)

        if self.cv_type == "s":
            weights = np.arange(1, len(rmsds) + 1)
            val = np.sum(weights * exps) / sum_exps
        elif self.cv_type == "z":
            val = -1.0 / self.lambda_param * np.log(sum_exps)

        assert np.isfinite(val), "Non-finite value detected."
        return float(val)

    def check_mdtraj_within_boundary(self, traj, milestone_variables, verbose=False, TOL=0.001):
        for frame_index in range(traj.n_frames):
            value = self.get_mdtraj_cv_value(traj, frame_index)
            result = self.check_value_within_boundary(value, milestone_variables, verbose=verbose, tolerance=TOL)
            if not result:
                return False
        return True

    def check_openmm_context_within_boundary(self, context, milestone_variables, positions=None, ref_positions=None, verbose=False, tolerance=0.0):
        value = self.get_openmm_context_cv_value(context, positions=positions, ref_positions=ref_positions, verbose=verbose, tolerance=tolerance)
        return self.check_value_within_boundary(value, milestone_variables, verbose=verbose, tolerance=tolerance)

    def check_value_within_boundary(self, value, milestone_variables, verbose=False, tolerance=0.0):
        milestone_k = milestone_variables["k"]
        milestone_value = milestone_variables["value"]
        if milestone_k * (value - milestone_value) > tolerance:
            if verbose:
                print(f"Path CV value ({value:.4f}) exceeded boundary ({milestone_value:.4f}).")
            return False
        return True

    def check_mdtraj_close_to_boundary(self, traj, milestone_variables, verbose=False, max_avg=0.03, max_std=0.05):
        diffs = []
        for frame_index in range(traj.n_frames):
            value = self.get_mdtraj_cv_value(traj, frame_index)
            milestone_value = milestone_variables["value"]
            diffs.append(value - milestone_value)

        avg_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if abs(avg_diff) > max_avg or std_diff > max_std:
            if verbose:
                print(f"Average diff: {avg_diff:.4f} nm, std: {std_diff:.4f} nm.")
            return False
        return True

    def get_atom_groups(self):
        return [self.group]

    def get_variable_values(self):
        return []


def make_mmvt_path_cv_object(path_cv_input, index, root_directory):
    """
    Helper function to parse XML input and create an MMVT_Path_CV object.
    """
    import shutil
    ref_file_basename = f"path_reference_cv_{index}.pdb"
    group = base.parse_xml_list(path_cv_input.group)
    align_group = base.parse_xml_list(path_cv_input.align_group) if hasattr(path_cv_input, 'align_group') else None
    
    absolute_ref_file = os.path.join(root_directory, ref_file_basename)
    shutil.copyfile(path_cv_input.ref_file, absolute_ref_file)
    
    cv = MMVT_Path_CV(
        index=index,
        group=group,
        ref_file=ref_file_basename,
        lambda_param=path_cv_input.lambda_param,
        cv_type=path_cv_input.cv_type,
        align_group=align_group
    )
    return cv
