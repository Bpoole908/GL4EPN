import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Union
from pdb import set_trace

import pandas as pd
import numpy as np

import gc4eptn
from gc4eptn.utils.utils import get_module_root, filter_df, recursive_getattr


class PMUData():
    data_path = get_module_root().parent / "datasets" 
    load_options = []
    topology_options = []
    def __init__(
        self,
        file_name,
        version,
        load,
        topology,
        n_pmu_features,
        pmus_per_edge,
        node_tags,
        generator_tags,
        transformer_tags,
        distribution_tags,
        voltage_tags,
        current_tags,
        parallel_current_tags,
        magnitude_tags,
        angle_tags,
        time_tag,
        vbase=1,
        ibase=1,
        vbase_gen=1, 
        ibase_gen=1, 
        vbase_dist=1,
        ibase_dist=1,
        column_mapping={},
        drop_current: bool = False,
        drop_parallel_currents: bool = False,
        drop_distribution: bool = False,
        drop_generators: bool = False,
        drop_magnitude: bool = False,
        drop_angle: bool = False,
    ):
        if not isinstance(load, list):
            assert load in self.load_options, f"`load` must be one of the following {self.load_options} but got {load}"
        if not isinstance(topology, list):
            assert topology in self.topology_options, f"`topology` must be one of the following {self.topology_options} but got {topology}"

        self.file_name = file_name
        self.version = version
        self.load = load
        self.topology = topology
        self.n_pmu_features = n_pmu_features
        self.pmus_per_edge = pmus_per_edge
        self.node_tags = node_tags
        self.generator_tags = generator_tags
        self.transformer_tags = transformer_tags
        self.distribution_tags = distribution_tags
        self.voltage_tags = voltage_tags
        self.current_tags = current_tags
        self.parallel_current_tags = parallel_current_tags
        self.magnitude_tags = magnitude_tags
        self.angle_tags = angle_tags
        self.time_tag = time_tag
        self.pu_bases = dict(
            vbase = vbase,
            ibase = ibase,
            vbase_gen = vbase_gen,
            ibase_gen = ibase_gen,
            vbase_dist = vbase_dist,
            ibase_dist = ibase_dist
        )
        self.column_mapping = column_mapping
        self.drop_current = drop_current
        self.drop_parallel_currents = drop_parallel_currents
        self.drop_distribution = drop_distribution
        self.drop_generators = drop_generators
        assert not (drop_magnitude and drop_angle), "Can only drop magnitude or angle, not both."
        self.drop_magnitude = drop_magnitude
        self.drop_angle = drop_angle
        if drop_magnitude:
            self._drop_pmu_features = self.magnitude_tags
            self.remaining_pmu_features =  self.n_pmu_features - 1
        elif drop_angle:
            self._drop_pmu_features = self.angle_tags
            self.remaining_pmu_features =  self.n_pmu_features - 1
        else:
            self._drop_pmu_features = None
            self.remaining_pmu_features =  self.n_pmu_features
        
        self.df = None
        self.graph_df = None
        self.nodes = set()
        self.node_info = dict()
        self.n_nodes = None
        self.n_features = None
        self.transmission = []
        self.generators = []
        self.distribution = []
        
    @property
    def file_path(self):
        file_paths = []
        load = self.load if isinstance(self.load, list) else [self.load]
            
        for l in load:
            file_path = self.data_path / self.version / self.topology / l / self.file_name
                
            if not file_path.exists():
                msg = f"Data a {file_path} data does not exist. Make sure it has " \
                    f"been downloaded and added to the pyccn/data/ directory."
                raise FileNotFoundError(msg)
            file_paths.append(file_path)
        
        return file_paths
    
    @abstractmethod
    def compute_ins_and_outs(self, df):
        pass
    
    def filter(self, df, regex):
        return filter_df(df, regex)
    
    def nested_filter(self, df, regex):
        if not isinstance(regex, (list, tuple)):
            regex = [regex]
        for re_list in regex:
            cols = self.filter(df, regex=re_list)
            if len(cols) != 0:
                df = df[cols]
            else:
                return np.array([], dtype=object)
             
        return np.array(list(df.columns))
    
    def groupby_regex(self, df, regex, sort=False, groups='extract', axis=1):
        if not isinstance(regex, (list, np.ndarray)):
            regex = [regex]
        if groups == 'extract':
            groups = [list(df.columns.str.extract(r, expand=False)) for r in regex]
        elif groups == 'replace':
            groups = [list(df.columns.str.replace(r, '', regex=True)) for r in regex]
        elif isinstance(groups, (list, np.ndarray)):
            groups = groups
        else:
            msg = "`groups` must be the strings 'extract' or 'replace' or groups given as a list."
            ValueError(msg)
            
        if axis == 0:
            return df.groupby(np.hstack(groups), sort=sort)
        elif axis == 1:
            return df.T.groupby(np.hstack(groups), sort=sort)
        else:
            msg = f"`axis` value {axis} is invalid, must be either 0 or 1."
            ValueError(msg)
    
    def group_node_edges(self, df):
        """ Group node edge features together 
        
            A given edge for any node will have a magnitude and angle measurement. These
            need to be grouped together so that each magnitude is mapped to its angle
            measurement.
        """
        edge_feature_names = self.magnitude_tags + self.angle_tags
        feature_regex = '|'.join([f for f in edge_feature_names])
        groups = self.groupby_regex(df, regex=feature_regex, groups='replace')
        return [tuple(grp.T) for _, grp in groups] 
    
    def get_node_value(self, nodes, key):
        if not isinstance(nodes, (list, tuple, np.ndarray)):
            nodes = [nodes] 
        assert len(self.node_info) != 0
        return [self.node_info[n][key] for n in nodes]
    

    def pu_norm(self, inplace=False):
        pu_bases = self.pu_bases
        norm_cols = [
            dict(
                cols=self.nested_filter(
                    self.df[self.transmission], 
                    [self.voltage_tags, self.magnitude_tags]
                ),
                base=pu_bases['vbase']
            ),
            dict(
                cols=self.nested_filter(
                    self.df[np.hstack([self.transmission, self.transformers])], 
                    [self.current_tags, self.magnitude_tags]
                ),
                base=pu_bases['ibase']
            ),
            dict(
                cols=self.nested_filter(
                    self.df[self.generators], 
                    [self.voltage_tags, self.magnitude_tags]
                ),
                base=pu_bases['vbase_gen']
            ),
            dict(
                cols=self.nested_filter(
                    self.df[self.generators], 
                    [self.current_tags, self.magnitude_tags]
                ),
                base=pu_bases['ibase_gen']
            ),
            dict(
                cols=self.nested_filter(
                    self.df[self.distribution], 
                    [self.voltage_tags, self.magnitude_tags]
                ),
                base=pu_bases['vbase_dist']
            ),
            dict(
                cols=self.nested_filter(
                    self.df[self.distribution], 
                    [self.current_tags, self.magnitude_tags]
                ),
                base=pu_bases['ibase_dist']
            ),
        ]
        
        if inplace:
            df = self.df
        else:
            df = self.df.copy()
            
        for c in norm_cols:
            # print(len(c['cols']), df.shape)
            if len(c['cols']) == 0:
                continue
            df[c['cols']] = df.loc[:, c['cols']].values / c['base']
        
        if not inplace:
            return df
        
    def load_data(self):
        """ Load basic PMU data using excel or csv files. """
        
        df = []
        for fp in self.file_path:
            ext = os.path.splitext(fp)[-1]
            if ext.startswith('.xl') or ext.startswith('.ol'):
                df.append(pd.read_excel(fp))
            else:
                df.append(pd.read_csv(fp))
        df = pd.concat(df, axis=0, ignore_index=True)
        df = self._drop_columns_and_compute_graph_info(df)
        self._count_nodes(df)
        self.df = df

        return self
        
    def _drop_columns_and_compute_graph_info(self, df):
        # Drop nodes we don't want to consider
        if self.drop_distribution:
            dist = self.filter(df, regex=self.distribution_tags)
            df = df[df.columns.drop(dist)]
        if self.drop_generators:
            gen = self.filter(df, regex=self.generator_tags+self.transformer_tags)
            df = df[df.columns.drop(gen)]
        if self.drop_parallel_currents:
            pc = self.filter(df, regex=self.parallel_current_tags)
            df = df[df.columns.drop(pc)]

        # Compute number of flows based on current connects, even if current features will be dropped
        self.flows = self.filter(df, regex=self.current_tags)
        # Number of pmus = Divide by number of pmu features
        self.n_pmus = int(len(self.flows) / self.n_pmu_features)
        assert (len(self.flows) / self.n_pmu_features) % self.n_pmu_features == 0
        # Number of edges = divide by two as we have 2 pmus per edge
        self.edges = int(self.n_pmus / self.pmus_per_edge)
        
        # Compute node related information
        self._compute_node_info(df)
        
        # Only drop all currents after computing graph information
        if self.drop_current:
            df = df[df.columns.drop(self.filter(df, regex=self.current_tags))]
            
        return df
    
    def _count_nodes(self, df):
        column_names = list(df.columns)                  
        found_nodes = []

        found_nodes = np.hstack(
                [re.findall(b, ' '.join(column_names)) for b in self.node_tags]
        )
        _, idx, self.n_features = np.unique(
            found_nodes, 
            return_index=True,
            return_counts=True,
        )
        # Simulate sorting based on idex of occurrence in raw data
        idx_idx = idx.argsort()
        self.nodes = found_nodes[idx[idx_idx]]
        self.n_features = self.n_features[idx_idx]
        self.n_nodes = len(self.nodes)

        # Parse different node types
        self.transformers = self.filter(df, self.transformer_tags)
        self.generators = self.filter(df, self.generator_tags) 
        self.distribution = self.filter(df, self.distribution_tags)
        
        nodes = self.filter(df, self.node_tags)
        used_nodes = np.hstack([self.generators, self.transformers, self.distribution])
        if len(used_nodes) != 0:
            self.transmission = np.setdiff1d(nodes, used_nodes)
        else:
            self.transmission = nodes
    
    def _compute_node_info(self, df):
        node_regex = f"({'|'.join([f for f in self.node_tags])})"
        node_groups = self.groupby_regex(df, node_regex)
        for name, grp in node_groups:
            # BUG: For some reason a nan group gets created
            if name == 'nan':
                continue
            grp = grp.T
            currents = self.filter(grp, regex=self.current_tags)
            voltage = self.filter(grp, regex=self.voltage_tags)
            if len(currents) != 0:
                n_ins, n_outs, n_unknown, ins, outs, unknown = self.compute_ins_and_outs(grp[currents])
                edges = n_ins+n_outs+n_unknown
            else:
                n_ins, n_outs, n_unknown, ins, outs, unknown, edges = [np.nan,]*7
            self.node_info[name] = dict(
                ins=ins,
                n_ins=n_ins,
                unknown=unknown,
                n_unknown=n_unknown,
                outs=outs,
                n_outs=n_outs,
                features=np.array(grp.columns),
                current=currents,
                voltage=voltage,
                edges=edges
            )
 
    def build_graph_data(
        self, 
        n: int = None, 
        norm_fn: Callable = None, 
        random: bool = False,
        rng: np.random.Generator = None,
    ):
        """ Loads raw data formatted as graph data
        
            Args:
                n: Size of window to be used. This window will only return a subset
                    of the data unless the length of the data is passed.
                    
                norm_fn: A reference to a function or a Callable object that takes in one
                    argument X and returns the norm of the data.
                    
        """

            
        if self.df is None:
            self.load_data()
        
        if norm_fn == 'pu':
            df = self.pu_norm()
        else:
            df = self.df.copy()
            
        if self.drop_magnitude or self.drop_angle:
            cols = self.filter(df, regex=self._drop_pmu_features)
            df = df[df.columns.drop(cols)]
            
        n = n or len(df)
        if random and n < len(df):
            if rng is None:
                rng = np.random.default_rng()
            idx = rng.choice(np.arange(len(df)), (n,), replace=False)
        else:
            idx = np.arange(n)
        ts = df[self.time_tag].iloc[idx]
        df = df.drop(self.time_tag, axis=1).iloc[idx]

        if norm_fn is not None and callable(norm_fn):
            df.loc[:] = norm_fn(df)

        #  Normalize data
        self.graph_df = self._reformat_raw_pmu(df, ts)
        
        return self


    def get_edge_locations(self):
        return np.where(self.A != 0)
    
    def _reformat_raw_pmu(self, df, ts):
        """ WARNING: Assumes node features are side-by-side """
       
        reshaped_dfs = []
        for n in self.nodes:
            node_df = df.filter(regex=(f"{n}(?!\d)"))
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
        
    def _generate_timestep_names(self, time_steps, columns_names):
        names = []
        for ts in time_steps:
            for col_name in columns_names:
                names.append(col_name + f"_{ts}")
        return names

  
class NovicePMUData(ABC):
    def data_checks(self):
        assert self.n_nodes == self.NODES, f"Wrong number of nodes {self.n_nodes} detected for `{type(self).__name__}`"
        assert self.df.shape[-1] in self.COLUMNS, f"Wrong number columns {self.df.shape[-1]} detected for `{type(self).__name__}`"
        assert self.edges in self.EDGES, f"Wrong number edges {self.edges} detected for `{type(self).__name__}`"
    
    @property
    @abstractmethod
    def graph_positions(self):
        pass
    
    @property
    @abstractmethod
    def true_flow_graph(self):
        pass
    
    @property
    @abstractmethod
    def true_network_graph(self):
        pass

class IntermediatePMUData(ABC):
    NODES = 13
    COLUMNS = [27, 75]
    EDGES = [12, 14]
    labels = {i:str(n)for i,n in enumerate(np.arange(1,14))}

    def data_checks(self):
        assert self.n_nodes == self.NODES, f"Wrong number of nodes {self.n_nodes} detected for `{type(self).__name__}`"
        assert self.df.shape[-1] in self.COLUMNS, f"Wrong number columns {self.df.shape[-1]} detected for `{type(self).__name__}`"
        assert self.edges in self.EDGES, f"Wrong number edges {self.edges} detected for `{type(self).__name__}`"

    @property
    @abstractmethod
    def graph_positions(self):
        pass
    
    @property
    @abstractmethod
    def true_flow_graph(self):
        pass
    
    @property
    @abstractmethod
    def true_network_graph(self):
        pass