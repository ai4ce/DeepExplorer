import numpy as np
from collections import defaultdict
import os
import json

# Initializing the Graph Class
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def addNode(self, value):
        self.nodes.add(value)

    def addEdge(self, fromNode, toNode, distance):
        self.edges[fromNode].append(toNode)
        self.distances[(fromNode, toNode)] = distance


class GraphNavigator(object):
    def __init__(self, nonconnect_dist = 1000. ):
        self.nonconnect_dist = nonconnect_dist

    # Implementing Dijkstra's Algorithm
    def dijkstra(self, graph, initial):
        visited = {initial: 0}
        path = defaultdict(list)

        nodes = set(graph.nodes)

        while nodes:
            minNode = None
            for node in nodes:
                if node in visited:
                    if minNode is None:
                        minNode = node
                    elif visited[node] < visited[minNode]:
                        minNode = node
            if minNode is None:
                break

            nodes.remove(minNode)
            currentWeight = visited[minNode]

            for edge in graph.edges[minNode]:
                weight = currentWeight + graph.distances[(minNode, edge)]
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge].append(minNode)

        return visited, path

    def compute_euclidean_dist(self, pos_list1, pos_list2):
        pos1 = np.array(pos_list1, np.float32)
        pos2 = np.array(pos_list2, np.float32)

        dist_val = np.sqrt(np.sum(np.square(pos1-pos2)))

        return dist_val

    def construct_graph(self, affinity_matrix_filename, nav_dict_filename ):
        customGraph = Graph()

        affinity_matrix = np.load(affinity_matrix_filename)
        with open(nav_dict_filename, 'r') as f:
            nav_dict = json.load(f)

        for action_id, action_name in enumerate( nav_dict['action_list'] ):
            start_obs_name = 'obs_{}'.format(action_id)
            end_obs_name = 'obs_{}'.format(action_id+1)
            dist_val = 0.3 if action_name == '1' else 0.
            customGraph.addNode(start_obs_name)
            customGraph.addNode(end_obs_name)
            customGraph.addEdge(start_obs_name, end_obs_name, distance=dist_val)
            customGraph.addEdge(end_obs_name, start_obs_name, distance=dist_val)

        for start_id in range(affinity_matrix.shape[0]):
            affinity_vec = affinity_matrix[start_id,:]
            node_id_2add = np.where(affinity_vec != self.nonconnect_dist )[0]
            if node_id_2add.shape[0] == 0:
                continue
            for end_id in node_id_2add:
                if end_id <= start_id:
                    continue
                dist_val = affinity_vec[end_id]
                real_dist_val = self.compute_euclidean_dist( nav_dict['position_list'][start_id],
                                                             nav_dict['position_list'][end_id] )
                if dist_val < real_dist_val and real_dist_val < 0.3:
                    dist_val = real_dist_val
                customGraph.addNode('obs_{}'.format(start_id))
                customGraph.addNode('obs_{}'.format(end_id))
                customGraph.addEdge('obs_{}'.format(start_id), 'obs_{}'.format(end_id), dist_val)
                customGraph.addEdge('obs_{}'.format(end_id), 'obs_{}'.format(start_id), dist_val)

        return customGraph

    def construct_a_node_from_imgname(self, input_imgname ):
        img_basename = os.path.basename(input_imgname)
        img_id = img_basename.split('_')[1].replace('.png','')
        nodename = 'obs_{}'.format(img_id)

        return nodename

    def find_shortest_path_4twonodes(self, custom_graph, start_node, end_node):
        visited, path = self.dijkstra(custom_graph, start_node)

        pathlen = visited[end_node]
        stepnum = len(path[end_node])

        return pathlen, stepnum


    def navigate_on_graph(self, custom_graph, episode_dict_filename ):
        with open(episode_dict_filename,'r') as f:
            episode_dict = json.load(f)

        for episode_id, episode_key in enumerate( episode_dict.keys() ):
            if episode_id%10 == 0:
                print('processed {}/{} episodes'.format(episode_id, len(episode_dict.keys())))
            start_imgname = episode_dict[episode_key]['start_pos_localized_img']
            end_imgname = episode_dict[episode_key]['end_pos_localized_img']
            start_nodename = self.construct_a_node_from_imgname(start_imgname)
            end_nodename = self.construct_a_node_from_imgname(end_imgname)

            pathlen, stepnum = self.find_shortest_path_4twonodes( custom_graph,
                                                                  start_nodename,
                                                                  end_nodename )

            episode_dict[episode_key]['navigation_result'] = dict()
            episode_dict[episode_key]['navigation_result']['pathlen'] = str(pathlen)
            episode_dict[episode_key]['navigation_result']['stepnum'] = str(stepnum)

        nav_result_savename = os.path.join( os.path.dirname(episode_dict_filename),
                                            os.path.basename(episode_dict_filename).replace('.json','_navresult.json'))

        with open(nav_result_savename, 'w', encoding='utf-8') as f:
            json.dump(episode_dict, f, ensure_ascii=False, indent=4)

        print('Done!')

    def navigate_on_two_graphs(self, custom_graph_floor0, custom_graph_floor1, episode_dict_filename ):
        with open(episode_dict_filename,'r') as f:
            episode_dict = json.load(f)

        for episode_id, episode_key in enumerate( episode_dict.keys() ):
            if episode_id%10 == 0:
                print('processed {}/{} episodes'.format(episode_id, len(episode_dict.keys())))
            start_imgname = episode_dict[episode_key]['start_pos_localized_img']
            end_imgname = episode_dict[episode_key]['end_pos_localized_img']
            floor_name = episode_dict[episode_key]['start_pos_floor']
            if floor_name == 'floor0':
                custom_graph = custom_graph_floor0
            else:
                custom_graph = custom_graph_floor1

            start_nodename = self.construct_a_node_from_imgname(start_imgname)
            end_nodename = self.construct_a_node_from_imgname(end_imgname)

            pathlen, stepnum = self.find_shortest_path_4twonodes( custom_graph,
                                                                  start_nodename,
                                                                  end_nodename )

            episode_dict[episode_key]['navigation_result'] = dict()
            episode_dict[episode_key]['navigation_result']['pathlen'] = str(pathlen)
            episode_dict[episode_key]['navigation_result']['stepnum'] = str(stepnum)

        nav_result_savename = os.path.join( os.path.dirname(episode_dict_filename),
                                            os.path.basename(episode_dict_filename).replace('.json','_navresult.json'))

        with open(nav_result_savename, 'w', encoding='utf-8') as f:
            json.dump(episode_dict, f, ensure_ascii=False, indent=4)

        print('Done!')


def main():
    ROOM_LIST = ['Cantwell', 'Denmark', 'Eastville', 'Edgemere', 'Elmira',
                 'Eudora','Greigsville', 'Pablo', 'Ribera', 'Sisters', 'Swormville']

    room_list_2floors = ['Mosquito','Sands','Scioto']

    ROOT_DIR = 'result_dir/'

    graph_navigator = GraphNavigator()


    compute_small_rooms = False
    compute_large_rooms = True
    if compute_small_rooms:
        for room_name_tmp in ROOM_LIST:
            print('Processing {}'.format(room_name_tmp))

            affinity_matrix_filename = os.path.join(ROOT_DIR,
                                                    room_name_tmp,
                                                    'affinity_matrix.npy')

            assert os.path.exists(affinity_matrix_filename)

            episode_info_filename = os.path.join( ROOT_DIR,
                                                  room_name_tmp,
                                                  '{}_reconnect_with_geodist.json'.format(room_name_tmp))

            nav_dict_filename = os.path.join(ROOT_DIR,
                                             room_name_tmp,
                                             '{}.json'.format(room_name_tmp))
            assert os.path.exists(nav_dict_filename)

            assert os.path.join(episode_info_filename)

            custom_graph = graph_navigator.construct_graph(affinity_matrix_filename,
                                                           nav_dict_filename )

            graph_navigator.navigate_on_graph( custom_graph, episode_info_filename )



    if compute_large_rooms:
        for room_name_tmp in room_list_2floors:
            print('processing the room: {}'.format(room_name_tmp))
            nav_dict_filename_floor0 = os.path.join(ROOT_DIR, room_name_tmp, '{}_floor0.json'.format(room_name_tmp))
            assert os.path.exists(nav_dict_filename_floor0)
            nav_dict_filename_floor1 = os.path.join(ROOT_DIR, room_name_tmp, '{}_floor1.json'.format(room_name_tmp))
            assert os.path.exists(nav_dict_filename_floor1)
            affinity_matrix_filename_floor0 = os.path.join(ROOT_DIR,
                                                    room_name_tmp,
                                                    'affinity_matrix_floor0.npy')

            affinity_matrix_filename_floor1 = os.path.join(ROOT_DIR,
                                                    room_name_tmp,
                                                    'affinity_matrix_floor1.npy')
            assert os.path.exists(affinity_matrix_filename_floor0)
            assert os.path.exists(affinity_matrix_filename_floor1)

            episode_info_filename = os.path.join( ROOT_DIR,
                                                  room_name_tmp,
                                                  '{}_reconnect_with_geodist.json'.format(room_name_tmp))

            assert os.path.exists(episode_info_filename)

            custom_graph_floor0 = graph_navigator.construct_graph(affinity_matrix_filename_floor0,
                                                                  nav_dict_filename_floor0)

            custom_graph_floor1 = graph_navigator.construct_graph(affinity_matrix_filename_floor1,
                                                                  nav_dict_filename_floor1)

            graph_navigator.navigate_on_two_graphs( custom_graph_floor0=custom_graph_floor0,
                                                    custom_graph_floor1=custom_graph_floor1,
                                                    episode_dict_filename=episode_info_filename )

    print('Done!')

if __name__ == '__main__':
    main()
