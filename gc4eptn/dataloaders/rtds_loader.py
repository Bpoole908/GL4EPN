import re
from typing import Callable, List, Union
from pdb import set_trace

import pandas as pd
import numpy as np

from gc4eptn.utils.utils import get_module_root
from gc4eptn.dataloaders import (
    PMUData,
    NovicePMUData,
    IntermediatePMUData,
)


class PMUDataRTDSV5(PMUData):
    data_path = get_module_root().parent / "datasets" / 'rtds'
    load_options =  ['low', 'medium', 'high']
    topology_options = ['complete', 'partial-left', 'partial-right']
    column_tag = "BUS{bus_num}_{feature_num}_{power_type}_{pmu_feature}"
    tag_delim = '_'
    
    def __init__(
        self,
        load: Union[List[str], str],
        topology: List[str],
        drop_current: bool = False,
        drop_parallel_currents: bool = False,
        drop_distribution: bool = False,
        drop_generators: bool = False,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
        no_flow_thresh: float = 0.001
    ):
        """
            When drop_parallel_currents=True and topology is set to left or right the
            removed edge corresponds to the edge that would be removed in the actual
            topology. With drop_parallel_currents=False and topology is set to left or right,
            the removed edge simply has a magnitude near 0.  
        """
        self.no_flow_thresh = no_flow_thresh
        super().__init__(
            file_name="Data_Raw_Complete.csv",
            version='v5',
            load=load,
            topology=topology,
            n_pmu_features=2,
            pmus_per_edge=2,
            generator_tags=['^BUS[1234](?!\d)'],
            transformer_tags=[
                'BUS5_02_I_ang', 'BUS5_02_I_mag', 'BUS6_02_I_ang', 'BUS6_02_I_mag',
                'BUS11_03_I_ang', 'BUS11_03_I_mag', 'BUS10_03_I_ang', 'BUS10_03_I_mag'
            ],
            node_tags=['BUS\d*'], 
            distribution_tags=[
                '^BUS[1][23](?!\d)', 'BUS7_03_I_ang', 'BUS7_03_I_mag', 
                'BUS9_05_I_ang', 'BUS9_05_I_mag'
            ],
            voltage_tags=['_V.*'],
            current_tags=['_I.*'],
            parallel_current_tags=[
                    'BUS7_04_I_ang', 'BUS7_04_I_mag', 'BUS8_04_I_ang', 'BUS8_04_I_mag',
                    'BUS8_02_I_ang', 'BUS8_02_I_mag', 'BUS9_02_I_ang', 'BUS9_02_I_mag'
            ],
            magnitude_tags=['mag'],
            angle_tags=['ang'],
            time_tag='Time',
            vbase=132790.5619,
            ibase=753.0655685,
            vbase_gen=11547.00538,
            ibase_gen=8660.254038,
            vbase_dist=132790.5619,
            ibase_dist=753.0655685,
            column_mapping={'BUS\d*_': ''},
            drop_current=drop_current,
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=drop_distribution,
            drop_generators=drop_generators,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
         
    def _reformat_raw_pmu(self, df, ts):
        if not self.drop_current:
            n_features = self.n_features
            if self.remaining_pmu_features == 1:
                n_features = self.n_features / self.n_pmu_features 
            max_feat_idx = np.argmax(n_features)
            max_feat = int(n_features[max_feat_idx])
            
        reshaped_dfs = []
        for n in self.nodes:
            node_df = df.filter(regex=(f"{n}(?!\d)"))
            
            # Set new column ID for identifying the number pmus. Each pmu can have 
            # the number of features equal to self.remaining_pmu_features.
            ids = np.arange(1, node_df.shape[-1]/self.remaining_pmu_features+1, dtype=int)
            if self.remaining_pmu_features == 1:
                col_id = ids
            else:
                col_id = np.zeros(node_df.shape[-1], dtype=int)
                col_id[0::2] = ids
                col_id[1::2] = ids
            node_df.columns = [re.sub('_\d*_', f'_{id:02d}_', col) for col, id in zip(node_df.columns, col_id)]
    
            if not self.drop_current and node_df.shape[-1] != max_feat:
                pad_pmus = (max_feat - node_df.shape[-1]) / self.remaining_pmu_features
                if self.drop_magnitude:
                    pad_pmu_tags = [self.angle_tags[0]] * int(pad_pmus)
                elif self.drop_angle:
                    pad_pmu_tags = [self.magnitude_tags[0]] * int(pad_pmus)
                else:
                    pad_pmu_tags = [self.angle_tags[0], self.magnitude_tags[0]] * int(pad_pmus)
                curr_feat_numb = [int(n.replace('_', '')) for n in node_df.columns.str.extract('(_\d*_)', expand=False)]
                missing_pmus =  int(max_feat/ self.remaining_pmu_features)
                col_numb = sorted(list(range(max(curr_feat_numb)+1, missing_pmus+1))*self.remaining_pmu_features)

                assert len(col_numb) == len(pad_pmu_tags), f"Node {n} {len(col_numb)} != {len(pad_pmu_tags)}"
                node_numb = node_df.columns[0].split('_')[0] 
                pad_pmu_cols = [f"{node_numb}_{n:02d}_I_{t}"for t, n in zip(pad_pmu_tags,col_numb)]
                columns = list(node_df.columns) + pad_pmu_cols
                node_df = node_df.reindex(columns=columns, fill_value=0, copy=False)
                
            node_data = node_df.values.ravel()
            # column_names = np.array([*node_df.columns]*len(node_data))
            column_names = self._generate_timestep_names(ts, list(node_df.columns))
            reshaped_node_df = pd.DataFrame(
                    node_data.reshape(1, -1), 
                    columns=column_names,
                    index=[n]
                )

            for key, value in self.column_mapping.items():
                reshaped_node_df.columns = reshaped_node_df.columns.str.replace(key, value, regex=True)
            reshaped_dfs.append(reshaped_node_df)

        return pd.concat(reshaped_dfs)
        
    def _generate_timestep_names(self, time_steps, column_names):
        names = []
        if isinstance(self.load, list):
            time_steps = np.arange(len(time_steps))
        for ts in time_steps:
            for col_name in column_names:
                names.append(col_name + f"_{ts}")
        return names

    def deconstruct_column_tag(self, tag):
        parts = tag.split(self.tag_delim)
        assert len(parts) == 4
        bus_num, feature_num, power_type, pmu_feature = parts
        bus_num = parts[0].replace('BUS', '')
        return bus_num, feature_num, power_type, pmu_feature
    

    
    # NOTE: This currently does not work for generator nodes due to how data is collected
    #       it is currently impossible to tell in/out for generator using RTDS dataset.
    def compute_ins_and_outs(self, df):
        current_mag_cols = self.filter(df, regex=self.magnitude_tags)
        no_flow = np.array(df[current_mag_cols].mean() < self.no_flow_thresh) 
        n_unknown = no_flow.sum()
        unknown_tag =  [self.tag_delim.join(self.deconstruct_column_tag(tag)[:-1]) for tag in current_mag_cols[no_flow]]
        unknown = self.filter(df, regex=unknown_tag)
        
        current_ang_cols = self.filter(df, regex=self.angle_tags)[~no_flow]
        current_ang = np.rad2deg(df[current_ang_cols])
        current_ang[current_ang < 0] += 360

        ins_check = (current_ang > 180).all(axis=0)
        ins_ang_tag = ins_check.index[ins_check == True].values
        ins_tag = [self.tag_delim.join(self.deconstruct_column_tag(tag)[:-1]) for tag in ins_ang_tag]
        ins = self.filter(df, regex=ins_tag)
        n_ins = int(len(ins) / self.n_pmu_features)

        outs_check = (current_ang < 180).all(axis=0)
        outs_ang_tag = outs_check.index[outs_check == True].values
        outs_tag = [self.tag_delim.join(self.deconstruct_column_tag(tag)[:-1]) for tag in outs_ang_tag]
        outs = self.filter(df, regex=outs_tag)
        n_outs = int(len(outs) / self.n_pmu_features)

        assert (len(ins) % self.n_pmu_features) == 0, "Number of pmu features does not divide number of ins correctly"
        assert (len(outs) % self.n_pmu_features) == 0, "Number of pmu features does not divide number of outs correctly"
        assert n_ins+n_outs+n_unknown == df.shape[-1] / self.n_pmu_features, "A PMU feature is being either not an in/out or is both"

        return n_ins, n_outs, n_unknown, ins, outs, unknown
    

class NovicePMUDataRTDSV5(PMUDataRTDSV5, NovicePMUData):
    NODES = 7
    COLUMNS = [15, 39, 47]
    EDGES = [6, 8]
    labels = {i:str(n)for i,n in enumerate(np.arange(5,12))}
    
    def __init__(
        self, 
        load: Union[List[str], str],
        topology: List[str] = 'complete',
        drop_current: bool = True, 
        drop_parallel_currents: bool = True,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataRTDSV5.__init__(
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
        
    @property
    def directed_arcs(self):
        if not self.drop_parallel_currents:
            arcs = [
                "arc3,rad=-.1",
                "angle3,angleA=89,angleB=-43",
                "arc3,rad=.15",
                "arc3,rad=-.25"if self.topology != 'partial-left' else None,
                "arc3,rad=.15",
                "arc3,rad=-.25" if self.topology != 'partial-right' else None,
                "angle3,angleA=95,angleB=35",
                "arc3,rad=.1",
            ]
            return [a for a in arcs if a is not None]
        else:
            return ["arc3,rad=0"]*self.edges
        
    @property
    def undirected_arcs(self):
        if not self.drop_parallel_currents:
            arcs = [
                "arc3,rad=-.1",
                "angle3,angleA=89,angleB=-43",
                "arc3,rad=.15",
                "arc3,rad=-.25"if self.topology != 'partial-left' else None,
                "arc3,rad=.15",
                "arc3,rad=-.25" if self.topology != 'partial-right' else None,
                "angle3,angleA=50,angleB=90",
                "arc3,rad=-.1",
            ]
            return [a for a in arcs if a is not None]
        else:
            return ["arc3,rad=0"]*self.edges
    
    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        A[0, 1] = 1
        A[1, 2] = 1
        A[2, 3] = 1 if self.drop_parallel_currents or self.topology == 'partial-left' else 2
        A[3, 4] = 1 if self.drop_parallel_currents or self.topology == 'partial-right' else 2
        A[5, 4] = 1 
        A[6, 5] = 1
        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        pos =  {
            0: np.array([0, .15]),
            1: np.array([1, .25]),
            2: np.array([2, .13]),
            3: np.array([3, .23]),
            4: np.array([4, .13]),
            5: np.array([5, .25]),
            6: np.array([6, .15]),
        }
        for k, v in pos.items():
            v[0] = v[0]*2
            v[1] = v[1]*32
        return pos
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self


class NoviceCurrentsPMUDataRTDSV5(NovicePMUDataRTDSV5):
     def __init__(
        self, 
        load: Union[List[str], str],
        topology: List[str] = 'complete',
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        super().__init__(
            load=load,
            topology=topology,
            drop_current=False, 
            drop_parallel_currents=False,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
 
      
class IntermediatePMUDataRTDSV5(PMUDataRTDSV5, IntermediatePMUData):
    def __init__(
        self, 
        load: Union[List[str], str],
        topology: List[str] = 'complete',
        drop_current: bool = True,
        drop_parallel_currents: bool = True, 
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataRTDSV5.__init__(
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
    
    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        # Generators
        A[0, 4] = 1
        A[1, 5] = 1
        A[2, 10] = 1
        A[3, 9] = 1
        # Transmission
        A[4, 5] = 1
        A[5, 6] = 1
        A[6, 7] = 1 if self.drop_parallel_currents or self.topology == 'partial-left' else 2
        A[7, 8] = 1 if self.drop_parallel_currents or self.topology == 'partial-right' else 2
        A[9, 8] = 1 
        A[10, 9] = 1
        # Distribution
        A[6, 11] = 1
        A[8, 12] = 1
        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        pos = {
            0: np.array([3, .33]),
            1: np.array([6, .40]),
            2: np.array([11, .33]),
            3: np.array([8, .40]),
            4: np.array([4, .15]),
            5: np.array([5, .30]),
            6: np.array([6, .13]),
            7: np.array([7, .33]),
            8: np.array([8, .13]),
            9: np.array([9, .30]),
            10: np.array([10, .15]),
            11: np.array([4.5, .03]),
            12: np.array([9.5, .03]),
        }
        for k, v in pos.items():
            v[0] = v[0]*2
            v[1] = v[1]*32
        return pos
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self
    
    
class IntermediateNoDistPMUDataRTDSV5(PMUDataRTDSV5, IntermediatePMUData):
    NODES = 11
    COLUMNS = [23, 63, 71]
    EDGES = [10, 12]
    labels = {i:str(n)for i,n in enumerate(np.arange(1,12))}
    
    def __init__(
        self, 
        load: Union[List[str], str],
        topology: List[str] = 'complete',
        drop_current: bool = True,
        drop_parallel_currents: bool = True, 
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataRTDSV5.__init__(
            self,
            load=load,
            topology=topology,
            drop_current=drop_current, 
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=True,
            drop_generators=False,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
        IntermediatePMUData.__init__(self)
    
    @property
    def true_flow_graph(self):
        A = np.zeros([self.NODES, self.NODES])
        # Generators
        A[0, 4] = 1
        A[1, 5] = 1
        A[2, 10] = 1
        A[3, 9] = 1
        # Transmission
        A[4, 5] = 1
        A[5, 6] = 1
        A[6, 7] = 1 if self.drop_parallel_currents else 2
        A[7, 8] = 1 if self.drop_parallel_currents else 2
        A[9, 8] = 1 
        A[10, 9] = 1
        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        pos = {
            0: np.array([3, .33]),
            1: np.array([6, .40]),
            2: np.array([11, .33]),
            3: np.array([8, .40]),
            4: np.array([4, .15]),
            5: np.array([5, .30]),
            6: np.array([6, .13]),
            7: np.array([7, .33]),
            8: np.array([8, .13]),
            9: np.array([9, .30]),
            10: np.array([10, .15]),
        }
        for k, v in pos.items():
            v[0] = v[0]*2
            v[1] = v[1]*4
        return pos
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self


class IntermediateNoGenPMUDataRTDSV5(PMUDataRTDSV5, IntermediatePMUData):
    NODES = 9
    COLUMNS = [19, 51, 59]
    EDGES = [6, 8, 10]
    labels = {i:str(n)for i,n in enumerate(np.arange(5,14))}
    
    def __init__(
        self, 
        load: Union[List[str], str],
        topology: List[str] = 'complete',
        drop_current: bool = True, 
        drop_parallel_currents: bool = True, 
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        PMUDataRTDSV5.__init__(
            self,
            load=load,
            topology=topology,
            drop_current=drop_current, 
            drop_parallel_currents=drop_parallel_currents,
            drop_distribution=False,
            drop_generators=True,
            drop_magnitude=drop_magnitude,
            drop_angle=drop_angle
        )
        IntermediatePMUData.__init__(self)
    
    @property
    def true_flow_graph(self):
        nodes = self.NODES
        A = np.zeros([nodes, nodes])
        # Transmission
        A[0, 1] = 1
        A[1, 2] = 1
        A[2, 3] = 1 if self.drop_parallel_currents else 2
        A[3, 4] = 1 if self.drop_parallel_currents else 2
        A[5, 4] = 1 
        A[6, 5] = 1
        # Distribution
        A[2, 7] = 1
        A[4, 8] = 1
        
        return A
    
    @property
    def true_network_graph(self):
        A = self.true_flow_graph
        return A.T + A
    
    @property
    def graph_positions(self):
        pos = {
            0: np.array([4, .15]),
            1: np.array([5, .30]),
            2: np.array([6, .13]),
            3: np.array([7, .33]),
            4: np.array([8, .13]),
            5: np.array([9, .30]),
            6: np.array([10, .15]),
            7: np.array([4.5, .03]),
            8: np.array([9.5, .03]),
        }
        for k, v in pos.items():
            v[0] = v[0]*2
            v[1] = v[1]*32
        return pos
    
    def load_data(self):
        super().load_data()
        self.data_checks()
        return self