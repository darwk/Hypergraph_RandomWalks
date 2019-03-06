from utils import write_hyperedges, write_map_to_disk


def get_file_paths(file):

    file_handle = open(file, 'r')

    file_list = []
    for file in file_handle.readlines():
        file_list.append(file[:-1])

    return file_list


def read_aminer_data_files(files_paths):

    # Set to store all referenced paper ids
    reference_ids = set([])

    # List to store referenced paper ids (dictionary storing hyperedges)
    references_list = []

    # Dictionary to store paper id to class id mapping
    paperid_classid = {}

    # Dictionary to store class id to class name mapping
    classid_classname = {}

    curr_classid = 1

    file_list = get_file_paths(files_paths)
    for file in file_list:

        classid_classname[curr_classid] = file

        file_handle = open(file, encoding="ISO-8859-1")     #TODO - dive deep about the encoding

        for line in file_handle.readlines():

            line_type = line[1]

            if line_type == 'i':
                index = int(line[6:-1])
                paperid_classid[index] = curr_classid
                references = set([])

            elif line_type == '%':
                if line[2] != ' ':
                    ref_index = int(line[2:-1])
                    references.add(ref_index)

            elif line_type == '!':
                if len(references) > 1:
                    references_list.append(references)
                    reference_ids.update(references)

        curr_classid += 1

    return paperid_classid, classid_classname, reference_ids, references_list


def validate_hyperedges(nodes, referenced_nodes, hyperedges):

    common_nodes = nodes & referenced_nodes
    valid_nodes = set([])

    i = 0
    while i < len(hyperedges):
        hyperedge = hyperedges[i]

        temp = hyperedge & common_nodes

        if len(temp) > 1:
            valid_nodes.update(temp)
            hyperedges[i] = list(temp)
            i += 1
        else:
            hyperedges.remove(hyperedge)

    return valid_nodes, hyperedges


def create_adjacencylist(hyperedges):
    adjlist = {}

    for hyperedge in hyperedges:
        for node1 in hyperedge:
            if node1 not in adjlist:
                adjlist[node1] = []

            for node2 in hyperedge:
                if node1 != node2:
                    adjlist[node1].append(node2)

    return adjlist


def get_citation_network(files_paths):
    
    paperid_classid, classid_classname, reference_ids, references_list = read_aminer_data_files(files_paths)

    nodes, hyperedges = validate_hyperedges(paperid_classid.keys(), reference_ids, references_list)

    paperid_classid = dict((node, paperid_classid[node]) for node in nodes)

    classid_count = {}

    for paperid in paperid_classid:
        classid = paperid_classid[paperid]
        if classid not in classid_count:
            classid_count[classid] = 1
        else:
            classid_count[classid] += 1

    for classid in classid_count:
        print("class - " + str(classid) + ", count - " + str(classid_count[classid]))

    print("nodes :", len(nodes))
    print("hyperedges :", len(hyperedges))
    write_map_to_disk("OutputFiles/paperid_classid.txt", paperid_classid)
    write_map_to_disk("OutputFiles/classid_classname.txt", classid_classname)
    write_hyperedges("OutputFiles/hyperedges.txt", hyperedges)
#    write_adjlist("OutputFiles/adjlist.txt", adjlist)

    return nodes, hyperedges, paperid_classid, classid_classname
