import networkx as nx


def get_file_paths(file):

    file_handle = open(file, 'r')

    file_list = []
    for file in file_handle.readlines():
        file_list.append(file[:-1])

    return file_list


def read_aminer_data_files():

    # Set to store all referenced paper ids
    reference_ids = set([])

    # List to store referenced paper ids (dictionary storing hyperedges)
    references_list = []

    # Dictionary to store paper id to class id mapping
    paperid_classid = {}

    # Dictionary to store class id to class name mapping
    classid_classname = {}

    curr_classid = 1

    file_list = get_file_paths("filePaths.txt")
    for file in file_list:

        classid_classname[curr_classid] = file

        file_handle = open(file, encoding="ISO-8859-1")     #TODO - dive deep about the encoding

        for line in file_handle.readlines():

            line_type = line[1]

            if line_type == 'i':
                index = line[6:-1]
                paperid_classid[index] = curr_classid
                references = set([])

            elif line_type == '%':
                if line[2] != ' ':
                    ref_index = line[2:-1]
                    references.add(ref_index)

            elif line_type == '!':
                if len(references) > 1:
                    references_list.append(references)
                    reference_ids.update(references)

        curr_classid += 1

    return paperid_classid, classid_classname, reference_ids, references_list


def read_citseer_data_files():

    # Set to store all referenced paper ids
    reference_ids = set([])

    # List to store referenced paper ids (dictionary storing hyperedges)
    references_list = []

    # Dictionary to store paper id to class id mapping
    paperid_classid = {}

    # Dictionary to store class id to class name mapping
    classid_classname = {}

    classname_classid = {}

    cites_file_handle = open("/home/darwin/datasets/Citeseer/citeseer/citeseer.cites")

    citations = {}
    paperids = set([])

    for line in cites_file_handle.readlines():
        temp = line.split("\n")
        link = temp[0].split("\t")

        if link[1] not in citations:
            citations[link[1]] = set([])
            paperids.add(link[1])

        if link[0] != link[1]:
            citations[link[1]].add(link[0])
            reference_ids.add(link[0])

    content_file_handle = open("/home/darwin/datasets/Citeseer/citeseer/citeseer.content")

    for line in content_file_handle.readlines():
        temp = line.split("\n")
        words = temp[0].split("\t")
        paperid = words[0]
        classname = words[-1]

        if classname not in classname_classid:
            classid = len(classname_classid)
            classname_classid[classname] = classid
            classid_classname[classid] = classname

        classid = classname_classid[classname]

        if paperid in paperids:
            if paperid not in paperid_classid:
                paperid_classid[paperid] = classid
            else:
                if paperid_classid[paperid] != classid:
                    print("Error - same paper id belongs to two different classes")

    for key in citations:
        if len(citations[key]) != 0:
            references_list.append(citations[key])

    return paperid_classid, classid_classname, reference_ids, references_list


def read_cora_data_files():

    # Set to store all referenced paper ids
    reference_ids = set([])

    # List to store referenced paper ids (dictionary storing hyperedges)
    references_list = []

    # Dictionary to store paper id to class id mapping
    paperid_classid = {}

    # Dictionary to store class id to class name mapping
    classid_classname = {}

    classname_classid = {}

    cites_file_handle = open("/home/darwin/datasets/Cora/cora/cora.cites")

    citations = {}
    paperids = set([])

    for line in cites_file_handle.readlines():
        temp = line.split("\n")
        link = temp[0].split("\t")

        if link[1] not in citations:
            citations[link[1]] = set([])
            paperids.add(link[1])

        if link[0] != link[1]:
            citations[link[1]].add(link[0])
            reference_ids.add(link[0])

    content_file_handle = open("/home/darwin/datasets/Cora/cora/cora.content")

    for line in content_file_handle.readlines():
        temp = line.split("\n")
        words = temp[0].split("\t")
        paperid = words[0]
        classname = words[-1]

        if classname not in classname_classid:
            classid = len(classname_classid)
            classname_classid[classname] = classid
            classid_classname[classid] = classname

        classid = classname_classid[classname]

        if paperid in paperids:
            if paperid not in paperid_classid:
                paperid_classid[paperid] = classid
            else:
                if paperid_classid[paperid] != classid:
                    print("Error - same paper id belongs to two different classes")

    for key in citations:
        if len(citations[key]) != 0:
            references_list.append(citations[key])

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


def get_largest_cc(nodes, hyperedges):
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for hyperedge in hyperedges:
        for i in range(len(hyperedge)):
            for j in range(i + 1, len(hyperedge)):
                G.add_edge(hyperedge[i], hyperedge[j])

    largest_cc = max(nx.connected_component_subgraphs(G), key=len)

    while i < len(hyperedges):
        hyperedge = hyperedges[i]
        temp = hyperedge & largest_cc.nodes()

        if len(temp) > 1:
            hyperedges[i] = list(temp)
            i += 1
        else:
            hyperedges.remove(hyperedge)

    return largest_cc.nodes(), hyperedges


def get_citation_network(dataset, use_cc):

    if dataset == "aminer":
        paperid_classid, classid_classname, reference_ids, references_list = read_aminer_data_files()
    elif dataset == "citeseer":
        paperid_classid, classid_classname, reference_ids, references_list = read_citseer_data_files()
    elif dataset == "cora":
        paperid_classid, classid_classname, reference_ids, references_list = read_cora_data_files()

    nodes, hyperedges = validate_hyperedges(paperid_classid.keys(), reference_ids, references_list)

    if use_cc == "True":
        nodes, hyperedges = get_largest_cc(nodes, hyperedges)

    print("Total number of nodes - " + str(len(nodes)))
    print("Total number of hyperedges - " + str(len(hyperedges)))

    paperid_classid = dict((node, paperid_classid[node]) for node in nodes)

    classid_count = {}
    for paperid in paperid_classid:
        classid = paperid_classid[paperid]
        if classid not in classid_count:
            classid_count[classid] = 1
        else:
            classid_count[classid] += 1

    for classid in classid_count:
        print("class - " + str(classid) + ", classname - " + str(classid_classname[classid]) + ", count - " + str(classid_count[classid]))

    return list(nodes), hyperedges, paperid_classid, classid_classname
