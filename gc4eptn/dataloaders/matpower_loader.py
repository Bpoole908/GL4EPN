from typing import Callable, List, Union
from pdb import set_trace

import numpy as np

from gc4eptn.utils.utils import get_module_root
from gc4eptn.dataloaders import (
    PMUData,
    NovicePMUData,
    IntermediatePMUData,
)

class PMUDataMatpowerCase9(PMUData):
    data_path = get_module_root().parent / "datasets" / 'matpower'
    # Large -> small angle indicates direction of flow
    load_options = ['80-90', '80-120', '100-110', '110-120']
    topology_options = ['complete']
    column_tag = "BUS{bus_num}_{feature_num}_{power_type}_{pmu_feature}"
    tag_delim = '_'
    
    def __init__(
        self,
        load: Union[List[str], str] = '80-120',
        topology: List[str] = 'complete',
        drop_current: bool = False,
        drop_parallel_currents: bool = False,
        drop_distribution: bool = False,
        drop_generators: bool = False,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUData.__init__(
            self,
            file_name="Data_pu_Complete.csv",
            version='case9',
            load=load,
            topology=topology,
            n_pmu_features=2,
            pmus_per_edge=2,
            generator_tags=['^BUS[123](?!\d)'],
            transformer_tags=[],
            node_tags=['BUS\d*'], 
            distribution_tags=[],
            voltage_tags=['_V.*'],
            current_tags=['_I.*'],
            parallel_current_tags=[],
            magnitude_tags=['mag'],
            angle_tags=['ang'],
            time_tag='Load Scales',
            # vbase=132790.5619,
            # ibase=753.0655685,
            # vbase_gen=11547.00538,
            # ibase_gen=8660.254038,
            # vbase_dist=132790.5619,
            # ibase_dist=753.0655685,
            column_mapping={'BUS\d*_': ''},
            drop_current=drop_current,
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=drop_distribution,
            drop_generators=drop_generators,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )

    def pu_norm(self, inplace=False):
        msg = "MatPower Data is already in PU format. This method is therefore not implemented."
        raise NotImplementedError(msg)

class NovicePMUDataMatpowerCase9(PMUDataMatpowerCase9, NovicePMUData):
    NODES = 6
    COLUMNS = [13]
    EDGES = [6]
    labels = {i:str(n)for i,n in enumerate(np.arange(4,10))}
    indices = {value:key for key, value in labels.items()} 
    
    def __init__(
        self, 
        load: Union[List[str], str] = '80-120',
        topology: List[str] = 'complete',
        drop_current: bool = True, 
        drop_parallel_currents: bool = True,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataMatpowerCase9.__init__(
            self,
            load=load,
            topology=topology,
            drop_current=drop_current, 
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=True,
            drop_generators=True,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
        NovicePMUData.__init__(self)
        
    def data_checks(self):
        assert self.n_nodes == self.NODES, f"Wrong number of nodes {self.n_nodes} detected for `{type(self).__name__}`"
        assert self.df.shape[-1] in self.COLUMNS, f"Wrong number columns {self.df.shape[-1]} detected for `{type(self).__name__}`"
        # assert self.edges in self.EDGES, f"Wrong number edges {self.edges} detected for `{type(self).__name__}`"
    
    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        # Transmission
        A[0, 1] = 1
        A[0, 5] = 1
        A[2, 1] = 1
        A[2, 3] = 1
        A[4, 3] = 1
        A[4, 5] = 1

        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        return {
            0: np.array([5,.10]),
            1: np.array([8,.20]),
            2: np.array([8,.30]),
            3: np.array([5,.25]),
            4: np.array([2,.30]),
            5: np.array([2,.20]),
        }
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self

        

class IntermediatePMUDataMatpowerCase9(PMUDataMatpowerCase9, IntermediatePMUData):
    NODES = 9
    COLUMNS = [10, 19]
    EDGES = [6, 9]
    labels = {i:str(n)for i,n in enumerate(np.arange(1,NODES+1))}
    indices = {value:key for key, value in labels.items()} 
    
    def __init__(
        self, 
        load: Union[List[str], str] = '80-120',
        topology: List[str] = 'complete',
        drop_current: bool = True, 
        drop_parallel_currents: bool = True, 
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataMatpowerCase9.__init__(
            self,
            load=load,
            topology=topology,
            drop_current=drop_current, 
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=False,
            drop_generators=False,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
        IntermediatePMUData.__init__(self)

    def data_checks(self):
        assert self.n_nodes == self.NODES, f"Wrong number of nodes {self.n_nodes} detected for `{type(self).__name__}`"
        assert self.df.shape[-1] in self.COLUMNS, f"Wrong number columns {self.df.shape[-1]} detected for `{type(self).__name__}`"
        # assert self.edges in self.EDGES, f"Wrong number edges {self.edges} detected for `{type(self).__name__}`"
    
    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        # Generators
        A[0, 3] = 1
        A[1, 7] = 1
        A[2, 5] = 1
        # Transmission
        A[3, 4] = 1
        A[3, 8] = 1
        A[5, 4] = 1
        A[5, 6] = 1
        A[7, 6] = 1
        A[7, 8] = 1
        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        return {
            0: np.array([3, 0.025]),
            1: np.array([-1,.25]),
            2: np.array([11,.25]),
            3: np.array([5,.10]),
            4: np.array([8,.20]),
            5: np.array([8,.30]),
            6: np.array([5,.25]),
            7: np.array([2,.30]),
            8: np.array([2,.20]),
        }
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self

class PMUDataMatpowerCase14(PMUData):
    data_path = get_module_root().parent / "datasets" / 'matpower'
    # Large -> small angle indicates direction of flow
    load_options = ['80-90', '80-120', '100-110', '110-120']
    topology_options = ['complete']
    column_tag = "BUS{bus_num}_{feature_num}_{power_type}_{pmu_feature}"
    tag_delim = '_'
    
    def __init__(
        self,
        load: Union[List[str], str] = '80-120',
        topology: List[str] = 'complete',
        drop_current: bool = False,
        drop_parallel_currents: bool = False,
        drop_distribution: bool = False,
        drop_generators: bool = False,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUData.__init__(
            self,
            file_name="Data_pu_Complete.csv",
            version='case14',
            load=load,
            topology=topology,
            n_pmu_features=2,
            pmus_per_edge=2,
            generator_tags=['^BUS[12368](?!\d)'],
            transformer_tags=[],
            node_tags=['BUS\d*'], 
            distribution_tags=[],
            voltage_tags=['_V.*'],
            current_tags=['_I.*'],
            parallel_current_tags=[],
            magnitude_tags=['mag'],
            angle_tags=['ang'],
            time_tag='Load Scales',
            # vbase=132790.5619,
            # ibase=753.0655685,
            # vbase_gen=11547.00538,
            # ibase_gen=8660.254038,
            # vbase_dist=132790.5619,
            # ibase_dist=753.0655685,
            column_mapping={'BUS\d*_': ''},
            drop_current=drop_current,
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=drop_distribution,
            drop_generators=drop_generators,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )

    def pu_norm(self, inplace=False):
        msg = "MatPower Data is already in PU format. This method is therefore not implemented."
        raise NotImplementedError(msg)

    
class NovicePMUDataMatpowerCase14(PMUDataMatpowerCase14, NovicePMUData):
    NODES = 9
    COLUMNS = [19]
    EDGES = []
    labels = {
        0:'4',
        1:'5',
        2:'7',
        3:'9',
        4:'10',
        5:'11',
        6:'12',
        7:'13',
        8:'14',
    }
    indices = {value:key for key, value in labels.items()} 
    
    def __init__(
        self, 
        load: Union[List[str], str] = '80-120',
        topology: List[str] = 'complete',
        drop_current: bool = True, 
        drop_parallel_currents: bool = True,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataMatpowerCase14.__init__(
            self,
            load=load,
            topology=topology,
            drop_current=drop_current, 
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=True,
            drop_generators=True,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
        NovicePMUData.__init__(self)
        
    def data_checks(self):
        assert self.n_nodes == self.NODES, f"Wrong number of nodes {self.n_nodes} detected for `{type(self).__name__}`"
        assert self.df.shape[-1] in self.COLUMNS, f"Wrong number columns {self.df.shape[-1]} detected for `{type(self).__name__}`"
        # assert self.edges in self.EDGES, f"Wrong number edges {self.edges} detected for `{type(self).__name__}`"

    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        # Transmission
        A[1, 0] = 1
        A[0, 2] = 1
        A[0, 3] = 1
        A[2, 3] = 1
        A[3, 4] = 1
        A[3, 8] = 1
        A[5, 4] = 1
        A[6, 7] = 1
        A[7, 8] = 1

        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        return {
            0: np.array([6,-2]),
            1: np.array([3,-2]),
            2: np.array([9, -0.5]),
            3: np.array([9, 2]),
            4: np.array([6.5, 3]),
            5: np.array([4, 3]),
            6: np.array([1, 4.5]),
            7: np.array([3, 6]),
            8: np.array([8, 4.75]),
        }
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self

        
class IntermediatePMUDataMatpowerCase14(PMUDataMatpowerCase14, IntermediatePMUData):
    NODES = 14
    COLUMNS = [29]
    EDGES = [20]
    labels = {i:str(n)for i,n in enumerate(np.arange(1,NODES+1))}
    indices = {value:key for key, value in labels.items()} 
    
    def __init__(
        self, 
        load: Union[List[str], str] = '80-120',
        topology: List[str] = 'complete',
        drop_current: bool = True, 
        drop_parallel_currents: bool = True, 
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataMatpowerCase14.__init__(
            self,
            load=load,
            topology=topology,
            drop_current=drop_current, 
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=False,
            drop_generators=False,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
        IntermediatePMUData.__init__(self)

    def data_checks(self):
        assert self.n_nodes == self.NODES, f"Wrong number of nodes {self.n_nodes} detected for `{type(self).__name__}`"
        assert self.df.shape[-1] in self.COLUMNS, f"Wrong number columns {self.df.shape[-1]} detected for `{type(self).__name__}`"
        # assert self.edges in self.EDGES, f"Wrong number edges {self.edges} detected for `{type(self).__name__}`"
    
    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        # Generators
        A[0, 1] = 1
        A[0, 4] = 1
        A[1, 2] = 1
        A[1, 3] = 1
        A[1, 4] = 1
        A[3, 2] = 1
        A[4, 5] = 1
        A[5, 10] = 1
        A[5, 11] = 1
        A[5, 12] = 1
        A[7, 6] = 1
        # Transmission
        A[4, 3] = 1
        A[3, 6] = 1
        A[3, 8] = 1
        A[6, 8] = 1
        A[8, 9] = 1
        A[8, 13] = 1
        A[10, 9] = 1
        A[11, 12] = 1
        A[12, 13] = 1
        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        return {
            0: np.array([0,0]),
            1: np.array([1,-5]),
            2: np.array([7,-5]),
            3: np.array([6,-2]),
            4: np.array([3,-2]),
            5: np.array([2.5, 0.5]),
            6: np.array([9, -0.5]),
            7: np.array([12, 0.5]),
            8: np.array([9, 2]),
            9: np.array([6.5, 3]),
            10: np.array([4, 3]),
            11: np.array([1, 4.5]),
            12: np.array([3, 6]),
            13: np.array([8, 4.75]),
        }
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self