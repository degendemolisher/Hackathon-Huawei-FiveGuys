def known_seeds(str = ''):
    seeds = [2381, 5351, 6047, 6829, 9221, 9859, 8053, 1097, 8677, 2521]
    # order by seed number
    seeds.sort()
    
    return seeds
print(known_seeds()) # [1097, 2381, 2521, 5351, 6047, 6829, 8053, 8677, 9221, 9859]