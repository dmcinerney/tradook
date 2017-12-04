


file_name = "images.txt"
with open(file_name) as f:
    info = open(file_name).read().split()
    all_names = [[None for _ in range(2)] for _ in range(len(info)/1)]
    for x in range(0,len(info)):
        all_names[x/2][x%2] = info[x]
print(all_names)
print(len(all_names))
