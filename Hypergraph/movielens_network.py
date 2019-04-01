from citation_network import validate_hyperedges, get_largest_cc


def read_movielens_data_files():
    movie_ids = set([])

    hyperedges = []

    actorid_movieids = {}

    movieid_genreid = {}

    genreid_genrename = {}
    genrename_genreid = {}

    dataset_folder = "/scratch/scratch5/darwk/Datasets/movielens"

    movies_file_handle = open(dataset_folder + "/movies.dat", encoding="ISO-8859-1")
    next(movies_file_handle)
    for line in movies_file_handle.readlines():
        temp = line.split("\n")
        line_split = temp[0].split("\t")

        movie_ids.add(line_split[0])

    actors_file_handle = open(dataset_folder + "/movie_actors.dat", encoding="ISO-8859-1")
    next(actors_file_handle)
    for line in actors_file_handle.readlines():
        temp = line.split("\n")
        line_split = temp[0].split("\t")

        movieid = line_split[0]
        actorid = line_split[1]

        if actorid not in actorid_movieids:
            actorid_movieids[actorid] = []

        actorid_movieids[actorid].append(movieid)

    genre_file_handle = open(dataset_folder + "/movie_genres.dat", encoding="ISO-8859-1")
    next(genre_file_handle)
    for line in genre_file_handle.readlines():
        temp = line.split("\n")
        line_split = temp[0].split("\t")

        movieid = line_split[0]
        genre = line_split[1]

        if genre not in genrename_genreid:
            genreid = len(genrename_genreid)
            genrename_genreid[genre] = genreid
            genreid_genrename[genreid] = genre

        genreid = genrename_genreid[genre]

        if movieid not in movieid_genreid:
            movieid_genreid[movieid] = []

        movieid_genreid[movieid].append(genreid)

    for actorid in actorid_movieids:
        hyperedges.append(actorid_movieids[actorid])

    return movieid_genreid, genreid_genrename, movie_ids, hyperedges


def get_movielens_network(use_cc):

    movieid_genreid, genreid_genrename, movie_ids, hyperedges = read_movielens_data_files()

    nodes, hyperedges = validate_hyperedges(movieid_genreid.keys(), movie_ids, hyperedges)

    if use_cc == "True":
        nodes, hyperedges = get_largest_cc(nodes, hyperedges)

    print("Total number of nodes - " + str(len(nodes)))
    print("Total number of hyperedges - " + str(len(hyperedges)))

    movieid_genreid = dict((node, movieid_genreid[node]) for node in nodes)

    genreid_count = {}
    for movieid in movieid_genreid:
        genreids = movieid_genreid[movieid]
        for genreid in genreids:
            if genreid not in genreid_count:
                genreid_count[genreid] = 1
            else:
                genreid_count[genreid] += 1

    for genreid in genreid_count:
        print("class - " + str(genreid) + ", classname - " + str(genreid_genrename[genreid]) + ", count - " + str(genreid_count[genreid]))

    return list(nodes), hyperedges, movieid_genreid, genreid_genrename
