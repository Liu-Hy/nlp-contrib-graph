# This file contains utility functions to parse the json files

# Find all the source sentences in a json file
def find_source(data, ls = []):
    if isinstance(data,dict):
        for key in data.keys():
            if key=="from sentence":
                sentences=data[key].split('\n') # The value can be one or more sentences
                for s in sentences:
                    ls.append(s.strip())
            elif isinstance(data[key], dict):
                find_source(data[key],ls)
            elif isinstance(data[key], list):
                for i in data[key]:
                    find_source(i,ls)
    elif isinstance(data,list):
        for i in data:
            find_source(i,ls)
    return ls    # might have repeated sentences

# determine if a triple is contained in the trace of traversing the json object in a depth-firsr manner
def is_contained(trace, triple):
    if len(trace) >= len(triple):
        for i in range(len(trace)-2):
            if trace[i:i+3] == triple:
                return True
        return False
    else:
        return False

# Get the previous two words in a trace,
# to be prefixed to the coordinated items in a list or a dictionary
def get_prefix(data, trace):
    if isinstance(data, dict) or isinstance(data, list):
        return trace[-2:]

# traverse the json object recursively,
# to find the source sentence of the triple
def find_tri_sent(data, triple, trace=[], ls=[], prefix=[]):
    # Parse the json file recursively and return a list of source sentences
    if isinstance(data, dict):
        for i, key in enumerate(data.keys()):
            if key != "from sentence":
                if prefix and i != 0:
                    trace += prefix
                trace.append(key)
                find_tri_sent(data[key], triple, trace, ls,
                              get_prefix(data[key], trace))
            else:
                if is_contained(trace, triple):
                    ls.append(data[key].strip())
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if prefix and i != 0:
                trace += prefix
            find_tri_sent(item, triple, trace, ls, prefix)
    elif isinstance(data, str):
        trace.append(data)
    return ls