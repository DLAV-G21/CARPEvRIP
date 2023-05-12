from typing import Any
import torch
from torch.nn.functional import softmax
import copy

import numpy as np

from utils.openpifpaf_helper import CAR_SKELETON_24, CAR_SKELETON_66 

class Decoder():
    
    def __init__(self, threshold=0.5, max_distance=100):
        self.threshold = threshold
        self.min_distance = max_distance

    def get_class_distribution_from_keypoints(self, keypoints):
        return softmax(keypoints[:,:,2:], dim=2)
    
    def get_class_distribution_from_links(self, links):
        return softmax(links[:,:,4:], dim=2)
    
    def get_class_from_distribution(self, distribution):
        return torch.topk(distribution, k=1, dim=2)
    
    def get_position_from_keypoints(self, keypoints):
        return keypoints[:,:,:2]
    
    def get_position_from_links(self, links):
        return links[:,:,:4]
    
    def get_bones_list(self, keypoints):
        if(keypoints.shape[2] - 3 == 24):
            return CAR_SKELETON_24, 24
        return CAR_SKELETON_66, 66

    def __call__(self, x, images_id=None):
        return self.forward(x, images_id)

    def forward(self, x, images_id=None):
        # Unpack the input
        keypoints, links = x

        # Get the keypoints class and probability
        keypoints_probability, keypoints_class = self.get_class_from_distribution(
            self.get_class_distribution_from_keypoints(
                keypoints
        ))
        # Get the keypoints positions
        keypoints_position = self.get_position_from_keypoints(
            keypoints
        )

        # Get the links class and probability
        links_probability, links_class = self.get_class_from_distribution(
            self.get_class_distribution_from_links(
                links
        ))
        # Get the links position
        links_position = self.get_position_from_links(
            links
        )

        # Get the bones list from the keypoints
        bones_list, nbr_keypoints = self.get_bones_list(keypoints)
        
        def get_dict(classs_, probabilitys_, positions_):
            # A dictionary to store the values
            dict_ = {}
            # Iterate through the three list using zip()
            for class_, probability_, position_ \
                in zip(classs_, probabilitys_, positions_):
                class_ = int(class_[0])
                probability_ = probability_[0]
                # Check if probability is greater than threshold and class is greater than zero
                if(probability_ >= self.threshold and class_ > 0):
                    # If the class already in the dictionary add the position and probability to the list
                    if class_ in dict_:
                        dict_[class_].append((position_, np.log(probability_)))
                    # Else create a new key with the position and probability
                    else:
                        dict_[class_] = [(position_, np.log(probability_))]
            # Return the dictionary
            return dict_
        
        def list_all_skeletons(keypoints):
            skeletons = []
            # For each keypoints
            for keypoints_class in keypoints:
                # For each v_id and keypoint in keypoints
                for keypoint_id, keypoint in enumerate(keypoints[keypoints_class]):
                    _, keypoint_probability = keypoint
                    # Add skeleton to skeletons
                    skeletons.append((keypoint_probability, {keypoints_class: keypoint_id}))
            # Return skeletons
            return skeletons

        def merge(bones_list, keypoints, links, skeletons):
            # For each link_class, keypoint_u_class, keypoint_v_class in bones_list
            for link_class, (keypoint_u_class, keypoint_v_class) in enumerate(bones_list):
                # If link_class exists in links and each keypoint_u_class and keypoint_v_class exists in keypoints
                if(link_class in links) and (keypoint_u_class in keypoints) and (keypoint_v_class in keypoints):
                    # For each link in the link_class
                    for link in links[link_class]:
                        # Get the link position and probability
                        link_position, link_probability = link
                        # Find the best keypoint u in keypoints with link position
                        best_keypoint_u = match(link_position[:2], keypoints[keypoint_u_class])
                        # Find the best keypoint v in keypoints with link position
                        best_keypoint_v = match(link_position[2:], keypoints[keypoint_v_class])

                        # If best keypoint u and v are found
                        if(best_keypoint_u >= 0) and (best_keypoint_v >= 0):
                            # List of skeletons with u
                            skeletons_with_keypoint_u = []
                            # List of skeletons with v
                            skeletons_with_keypoint_v = []
                            # List of skeletons with u and v
                            skeletons_with_keypoint_u_v = []

                            # For each skeleton in skeletons
                            for id_, skeleton in enumerate(skeletons):
                                keypoint_u_in = keypoint_u_class in skeleton[1]
                                keypoint_v_in = keypoint_v_class in skeleton[1]
                                best_keypoint_u_in = keypoint_u_in and (skeleton[1][keypoint_u_class] == best_keypoint_u)
                                best_keypoint_v_in = keypoint_v_in and (skeleton[1][keypoint_v_class] == best_keypoint_v)

                                # If best_u and best_v already in skeleton
                                if best_keypoint_u_in and best_keypoint_v_in:
                                    # Add id_ to skeletons_with_keypoint_u_v
                                    skeletons_with_keypoint_u_v.append(id_)
                                # If best_u already in skeleton and no keypoint of type v is in skeleton
                                elif best_keypoint_u_in and (not keypoint_v_in):
                                    # Add id_ to skeletons_with_keypoint_u
                                    skeletons_with_keypoint_u.append(id_)
                                # If best_v already in skeleton and no keypoint of type u is in skeleton
                                elif best_keypoint_v_in and (not keypoint_u_in):
                                    # Add id_ to skeletons_with_keypoint_v
                                    skeletons_with_keypoint_v.append(id_)

                            # For each skeleton_with_u
                            for skeletons_with_keypoint_u_id in skeletons_with_keypoint_u:
                                # For each skeleton_with_v
                                for skeletons_with_keypoint_v_id in skeletons_with_keypoint_v:
                                    # If skeletons_with_u and skeletons_with_v have no common keys
                                    if(len(set(skeletons[skeletons_with_keypoint_u_id][1].keys()).intersection(set(skeletons[skeletons_with_keypoint_v_id][1].keys()))) == 0):
                                        # New skeleton's bones is combination of bones of skeletons[skeletons_with_keypoint_u_id] and skeletons[skeletons_with_keypoint_v_id]
                                        new_dict = copy.deepcopy(skeletons[skeletons_with_keypoint_u_id][1])
                                        new_dict.update(skeletons[skeletons_with_keypoint_v_id][1])
                                        # Add new skeleton to skeletons
                                        skeletons_with_keypoint_u_v.append(len(skeletons))
                                        skeletons.append((
                                            # Log probability of new skeleton
                                            link_probability + skeletons[skeletons_with_keypoint_u_id][0] + skeletons[skeletons_with_keypoint_v_id][0],
                                            new_dict
                                        ))

                            skeletons_to_remove = set()
                            for skeletons_1_id in skeletons_with_keypoint_u_v:
                                for skeletons_2_id in skeletons_with_keypoint_u_v:
                                    if(skeletons_1_id < skeletons_2_id):
                                        if(skeletons[skeletons_1_id][1] == skeletons[skeletons_2_id][1]):
                                            if(skeletons[skeletons_1_id][0] > skeletons[skeletons_2_id][0]):
                                                skeletons_to_remove.add(skeletons_2_id)
                                            else:
                                                skeletons_to_remove.add(skeletons_1_id)

                            skeletons = [s for i,s in enumerate(skeletons) if (i not in skeletons_to_remove)]

            # Return all skeletons
            return skeletons

        def match(link_position, keypoints):
            # Set min_distance to the class's min_distance
            min_distance = self.min_distance
            # Set best to -1, meaning no best match has been found yet
            best = -1

            # Iterate through all the keypoints
            for i, keypoint in enumerate(keypoints):
                # Get the keypoint's position
                keypoint_position, _ = keypoint
                # Calculate the distance between the link position and the keypoint
                distance = np.sqrt(np.sum(abs(keypoint_position - link_position)))

                # Check if the distance is less than the min_distance
                if(distance < min_distance):
                    # If it is, set best to the current keypoint's index
                    best = i
                    # And set min_distance to the current distance
                    min_distance = distance
            
            # Return the index of the best match
            return best
        
        def filter(skeletons, keypoints, image_id):
            # List to store the filtered skeletons
            detrected_elements = []
            # Set to store the used keypoints
            used_keypoints = {}
            # Variable to keep track of the id
            id = 1
            # Iterate over the skeletons sorted by their length and score
            for (skeleton_probability, skeleton) in sorted(skeletons, key=lambda x: (len(x[1]), x[0]), reverse=True):
                # Check if the keypoints of this skeleton have already been used
                if(all([(k not in used_keypoints) or (skeleton[k] not in used_keypoints[k]) for k in skeleton])):
                    # Array to store the coordinates of each keypoint
                    keypoints_ = np.zeros(nbr_keypoints*3)

                    # Iterate over the different keypoints
                    for k in skeleton:
                        # Add the used keypoint to the set
                        if(k not in used_keypoints):
                            used_keypoints[k] = set()
                        used_keypoints[k].add(skeleton[k])

                        # Get the x and y coordinates of the keypoint
                        keypoint_position, _ = keypoints[k][skeleton[k]]
                        # Store the coordinates
                        keypoints_[(k-1)*3] = keypoint_position[0]
                        keypoints_[(k-1)*3+1] = keypoint_position[1]
                        keypoints_[(k-1)*3+2] = 2.0


                    # Create the element object
                    element = {
                        'image_id' : image_id,
                        'score' : np.exp(skeleton_probability),
                        'category_id': 1,
                        'iscrowd': 0,
                        'id': int(str(image_id)+str(id)), #ecivalen XD <3 : image_id*10**math.ceil(np.log10(id+1))+id
                        'bbox': [0,0,0,0],
                        'num_keypoints': len(skeleton),
                        'keypoints': keypoints_,
                        'segmentation':[],
                    }

                    # Append the element object to the list of detrected_elements
                    detrected_elements.append(element)

            # Return the filtered skeletons
            return detrected_elements

        keypoints_class = keypoints_class.to('cpu').numpy()
        keypoints_probability = keypoints_probability.to('cpu').numpy()
        keypoints_position = keypoints_position.to('cpu').numpy()
        links_class = links_class.to('cpu').numpy()
        links_probability = links_probability.to('cpu').numpy()
        links_position = links_position.to('cpu').numpy()

        result = []
        for b in range(keypoints.shape[0]):

            # Get dict for keypoints and links
            keypoints = get_dict(keypoints_class[b], keypoints_probability[b], keypoints_position[b])
            links = get_dict(links_class[b], links_probability[b], links_position[b])

            # Generate all possible skeletons with one keypoint
            skeletons = list_all_skeletons(keypoints)

            # Merge skeletons with bones_list
            skeletons = merge(bones_list, keypoints, links, skeletons)

            image_id = b if images_id is None else images_id[b]

            # Filter out skeletons with wrong number of keypoints
            detrected_skeletons = filter(skeletons, keypoints, image_id)

            if len(detrected_skeletons) <= 0:
                # Create the element object
                element = {
                    'image_id' : image_id,
                    'score' : 0,
                    'category_id': 1,
                    'iscrowd': 0,
                    'id': int(str(image_id)+str(id)), #ecivalen XD <3 : image_id*10**math.ceil(np.log10(id+1))+id
                    'bbox': [0,0,0,0],
                    'num_keypoints': 0,
                    'keypoints': np.zeros(nbr_keypoints*3),
                    'segmentation':[],
                }

                # Append the element object to the list of detrected_elements
                detrected_skeletons.append(element)

            # Add the skeletons to the result
            result.extend(detrected_skeletons)

        # Return the result
        return result
