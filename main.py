from dataclasses import dataclass, field
from typing import Any
import json

nodes = []
obs_weight = 0.1

def clean(dataset_path):
    global dataset
    with open(dataset_path, "r") as f:
        return json.load(f)
    
@dataclass
class Node:
    name: str
    type: str
    connections: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls, 
        name: str, 
        node_type: str, 
        connections: dict[str, Any] | None = None, 
        data: dict[str, Any] | None = None
    ) -> "Node":
        return cls(
            name=name,
            type=node_type,
            connections=connections if connections is not None else {},
            data=data if data is not None else {}
        )

# builds huge database of empty nodes
def build_memory(dataset):
    global nodes
    
    unique_words = []
    
    for sentence in dataset:
        for word in sentence:
            unique_words.append[word]
            
    # remove duplicates
    unique_words = set(unique_words)
    unique_words = list(unique_words)
    
    for word in unique_words:
        current_word = Node.create(
            word,
            "", # replace later with learned type
            connections={},
            data={"dopamine": 0}
        )        

def direct(*args): # a = {"a": ["stimulus", float(outcome_score), float(dopamine)]} e.g. {"dog": ["sight", 0.5, 0]}
    if len(args) == 1:
        key, data = next(iter(a.items()))
        if data[0] == "sight":
            observational(key, data)
        else:
            operant(key, data)
    else:
        classical(args)
        
def classical(*args): # a/b = {"a": "stimulus", "b": "stimulus"} e.g. {"dog": "sight", "bark": "sound"}
    # if fired at the same time, make connection thicker
    for arg in args:
        key, data = next(iter(a.items()))
        if not key.connections:
            for curr_arg in args:
                if curr_arg == arg: continue
                else:
                    arg.connections[arg] = 1
        else:
            # just add 0.25 for now
            for curr_arg in args:
                if curr_arg == arg: continue
                else:
                    arg.connections[arg] += 0.25
                    
def operant(key, data):
    outcome_score = perform(key) # outcome score based on positive feedback from result
    if outcome_score > data[1]:
        # add difference to existing dopamine
        key.data[dopamine] += outcome_score - data[1]
    elif outcome_score < data[1]:
        key.data[dopamine] -= outcome_score - data[1]
        # if equal do nothing
        
def observational(key, data):
    # same as operant; simulate action, get rewarded but less due to others performing action. makes you want to copy if they get a good result
    outcome_score = perform(key) # outcome score based on positive feedback from result
    if outcome_score > data[1]:
        # add difference to existing dopamine
        key.data[dopamine] += (outcome_score - data[1]) * obs_weight
    elif outcome_score < data[1]:
        key.data[dopamine] -= (outcome_score - data[1]) * obs_weight
        # if equal do nothing

# node = Node.create(
#     "Signal", 
#     "event", 
#     connections={"pan": 1, "warm": 1.25}, 
#     data={"dopamine": 2, "decay": 1}
# )

dataset = clean("dataset.txt")
