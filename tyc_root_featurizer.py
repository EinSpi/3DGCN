
import inspect
import logging
import numpy as np
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from typing import List, Tuple

from deepchem.utils import get_print_threshold

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData

from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot

logger = logging.getLogger(__name__)


class tyc_root_Featurizer(object):
  """Abstract class for calculating a set of features for a datapoint.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_featurize` method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
  """

  def featurize(self,
                datapoints: Iterable[Any],
                log_every_n: int = 1000,
                **kwargs) -> np.ndarray:
    """Calculate features for datapoints.

    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.

    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    for i, point in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        features.append(self._featurize(point, **kwargs))
      except:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))
    #modified by tyc, original: return np.asarray(features)
    return np.asarray(features)

  def __call__(self, datapoints: Iterable[Any], **kwargs):
    """Calculate features for datapoints.

    `**kwargs` will get passed directly to `Featurizer.featurize`

    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    #tyc mmodified here for the _ before featurize(),originally without_
    return self.featurize(datapoints, **kwargs)

  def _featurize(self, datapoint: Any, **kwargs):
    """Calculate features for a single datapoint.

    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __repr__(self) -> str:
    """Convert self to repr representation.

    Returns
    -------
    str
      The string represents the class.

    Examples
    --------
    >>> import deepchem as dc
    >>> dc.feat.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False, is_counts_based=False]
    >>> dc.feat.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_info = ''
    for arg_name in args_names:
      value = self.__dict__[arg_name]
      # for str
      if isinstance(value, str):
        value = "'" + value + "'"
      # for list
      if isinstance(value, list):
        threshold = get_print_threshold()
        value = np.array2string(np.array(value), threshold=threshold)
      args_info += arg_name + '=' + str(value) + ', '
    return self.__class__.__name__ + '[' + args_info[:-2] + ']'

  def __str__(self) -> str:
    """Convert self to str representation.

    Returns
    -------
    str
      The string represents the class.

    Examples
    --------
    >>> import deepchem as dc
    >>> str(dc.feat.CircularFingerprint(size=1024, radius=4))
    'CircularFingerprint_radius_4_size_1024'
    >>> str(dc.feat.CGCNNFeaturizer())
    'CGCNNFeaturizer'
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_num = len(args_names)
    args_default_values = [None for _ in range(args_num)]
    if args_spec.defaults is not None:
      defaults = list(args_spec.defaults)
      args_default_values[-len(defaults):] = defaults

    override_args_info = ''
    for arg_name, default in zip(args_names, args_default_values):
      if arg_name in self.__dict__:
        arg_value = self.__dict__[arg_name]
        # validation
        # skip list
        if isinstance(arg_value, list):
          continue
        if isinstance(arg_value, str):
          # skip path string
          if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
            continue
        # main logic
        if default != arg_value:
          override_args_info += '_' + arg_name + '_' + str(arg_value)
    return self.__class__.__name__ + override_args_info
  
class tyc_MolecularFeaturizer(tyc_root_Featurizer):
  

  def __init__(self, use_original_atoms_order=False):
    """
    Parameters
    ----------
    use_original_atoms_order: bool, default False
      Whether to use original atom ordering or canonical ordering (default)
    """
    self.use_original_atoms_order = use_original_atoms_order

  def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
    """Calculate features for molecules.

    Parameters
    ----------
    datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    if 'molecules' in kwargs:
      datapoints = kwargs.get("molecules")
      raise DeprecationWarning(
          'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
      )

    # Special case handling of single molecule
    if isinstance(datapoints, str) or isinstance(datapoints, Mol):
      datapoints = [datapoints]
    else:
      # Convert iterables to list
      datapoints = list(datapoints)

    features: list = []
    
    for i, mol in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:      
        if isinstance(mol, str):
          # condition if the original atom order is required
          if hasattr(
              self,
              'use_original_atoms_order') and self.use_original_atoms_order:
            # mol must be a RDKit Mol object, so parse a SMILES
            mol = Chem.MolFromSmiles(mol)
          else:
            # mol must be a RDKit Mol object, so parse a SMILES
            mol = Chem.MolFromSmiles(mol)
            # SMILES is unique, so set a canonical order of atoms
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        kwargs_per_datapoint = {}
        for key in kwargs.keys():
          kwargs_per_datapoint[key] = kwargs[key][i]
        features.append(self._featurize(mol, **kwargs_per_datapoint))
      except Exception as e:
        if isinstance(mol, Chem.rdchem.Mol):
          mol = Chem.MolToSmiles(mol)
        logger.warning(
            "Failed to featurize datapoint %d, %s. Because it has just 1 atom. Appending fake array", i,
            mol)
        features.append(GraphData(node_features=np.asarray([[1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0]]),
                     edge_index=np.asarray([[0,1],[1,0]]),
                     edge_features=None,
                     node_pos_features=np.asarray([[0,0,0],[0,0,1]])))
        #features.append(np.array([]))

    
    
    
    #modifyied by tyc, originally np.asarray(features)

    return np.asarray(features)
  
class tyc_MolGraphConvFeaturizer(tyc_MolecularFeaturizer):
  """This class is a featurizer of general graph convolution networks for molecules.

  The default node(atom) and edge(bond) representations are based on
  `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
  you could use this class as a guide to define your original Featurizer. In many cases, it's enough
  to modify return values of `construct_atom_feature` or `construct_bond_feature`.

  The default node representation are constructed by concatenating the following values,
  and the feature length is 30.

  - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
  - Formal charge: Integer electronic charge.
  - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
  - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
  - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
  - Degree: A one-hot vector of the degree (0-5) of this atom.
  - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
  - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
  - Partial charge: Calculated partial charge. (Optional)

  The default edge representation are constructed by concatenating the following values,
  and the feature length is 11.

  - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
  - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
  - Conjugated: A one-hot vector of whether this bond is conjugated or not.
  - Stereo: A one-hot vector of the stereo configuration of a bond.

  If you want to know more details about features, please check the paper [1]_ and
  utilities in deepchem.utils.molecule_feature_utils.py.

  Examples
  --------
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
  >>> out = featurizer.featurize(smiles)
  >>> type(out[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> out[0].num_node_features
  30
  >>> out[0].num_edge_features
  11

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
     Journal of computer-aided molecular design 30.8 (2016):595-608.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               use_edges: bool = False,
               use_chirality: bool = False,
               use_partial_charge: bool = False):
    """
    Parameters
    ----------
    use_edges: bool, default False
      Whether to use edge features or not.
    use_chirality: bool, default False
      Whether to use chirality information or not.
      If True, featurization becomes slow.
    use_partial_charge: bool, default False
      Whether to use partial charge data or not.
      If True, this featurizer computes gasteiger charges.
      Therefore, there is a possibility to fail to featurize for some molecules
      and featurization becomes slow.
    """
    self.use_edges = use_edges
    self.use_partial_charge = use_partial_charge
    self.use_chirality = use_chirality

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
    """Calculate molecule graph features from RDKit mol object.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit mol object.

    Returns
    -------
    graph: GraphData
      A molecule graph with some features.
    """
    assert datapoint.GetNumAtoms(
    ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )

    if self.use_partial_charge:
      try:
        datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
      except:
        # If partial charges were not computed
        try:
          from rdkit.Chem import AllChem
          AllChem.ComputeGasteigerCharges(datapoint)
        except ModuleNotFoundError:
          raise ImportError("This class requires RDKit to be installed.")

    # construct atom (node) feature
    h_bond_infos = construct_hydrogen_bonding_info(datapoint)
    atom_features = np.asarray(
        [
            _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                    self.use_partial_charge)
            for atom in datapoint.GetAtoms()
        ],
        dtype=float,
    )

    # construct edge (bond) index
    src, dest = [], []
    for bond in datapoint.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dest += [end, start]

    # construct edge (bond) feature
    bond_features = None  # deafult None
    if self.use_edges:
      features = []
      for bond in datapoint.GetBonds():
        features += 2 * [_construct_bond_feature(bond)]
      bond_features = np.asarray(features, dtype=float)

    # load_sdf_files returns pos as strings but user can also specify
    # numpy arrays for atom coordinates
    pos = []
    if 'pos_x' in kwargs and 'pos_y' in kwargs and 'pos_z' in kwargs:
      if isinstance(kwargs['pos_x'], str):
        pos_x = eval(kwargs['pos_x'])
      elif isinstance(kwargs['pos_x'], np.ndarray):
        pos_x = kwargs['pos_x']
      if isinstance(kwargs['pos_y'], str):
        pos_y = eval(kwargs['pos_y'])
      elif isinstance(kwargs['pos_y'], np.ndarray):
        pos_y = kwargs['pos_y']
      if isinstance(kwargs['pos_z'], str):
        pos_z = eval(kwargs['pos_z'])
      elif isinstance(kwargs['pos_z'], np.ndarray):
        pos_z = kwargs['pos_z']

      for x, y, z in zip(pos_x, pos_y, pos_z):
        pos.append([x, y, z])

      
      

    return GraphData(node_features=atom_features,
                     edge_index=np.asarray([src, dest], dtype=int),
                     edge_features=bond_features,
                     node_pos_features=np.asarray(pos))
  
def _construct_atom_feature(atom: RDKitAtom, h_bond_infos: List[Tuple[int,
                                                                      str]],
                            use_chirality: bool,
                            use_partial_charge: bool) -> np.ndarray:
  """Construct an atom feature from a RDKit atom object.

  Parameters
  ----------
  atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
  h_bond_infos: List[Tuple[int, str]]
    A list of tuple `(atom_index, hydrogen_bonding_type)`.
    Basically, it is expected that this value is the return value of
    `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
    value is "Acceptor" or "Donor".
  use_chirality: bool
    Whether to use chirality information or not.
  use_partial_charge: bool
    Whether to use partial charge data or not.

  Returns
  -------
  np.ndarray
    A one-hot vector of the atom feature.
  """
  atom_type = get_atom_type_one_hot(atom)
  formal_charge = get_atom_formal_charge(atom)
  hybridization = get_atom_hybridization_one_hot(atom)
  acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
  aromatic = get_atom_is_in_aromatic_one_hot(atom)
  degree = get_atom_total_degree_one_hot(atom)
  total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
  atom_feat = np.concatenate([
      atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
      total_num_Hs
  ])

  if use_chirality:
    chirality = get_atom_chirality_one_hot(atom)
    atom_feat = np.concatenate([atom_feat, np.array(chirality)])

  if use_partial_charge:
    partial_charge = get_atom_partial_charge(atom)
    atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
  return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
  """Construct a bond feature from a RDKit bond object.

  Parameters
  ---------
  bond: rdkit.Chem.rdchem.Bond
    RDKit bond object

  Returns
  -------
  np.ndarray
    A one-hot vector of the bond feature.
  """
  bond_type = get_bond_type_one_hot(bond)
  same_ring = get_bond_is_in_same_ring_one_hot(bond)
  conjugated = get_bond_is_conjugated_one_hot(bond)
  stereo = get_bond_stereo_one_hot(bond)
  return np.concatenate([bond_type, same_ring, conjugated, stereo])


