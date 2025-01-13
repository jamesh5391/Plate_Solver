import struct
import os 

def parse_star_record(record):
    """
    Parse a 5-byte or 6-byte star record and return RA, DEC, and optional magnitude.
    """
    ra_bytes = record[0:3]
    dec_bytes = record[3:5]
    
    
    ra = int.from_bytes(ra_bytes, byteorder='little')
    dec = int.from_bytes(dec_bytes, byteorder='little', signed=True)
    
    # Convert to hours and degrees
    ra_hours = ra * 24 / ((256**3) - 1)
    dec_degrees = dec * 90 / ((128*256*256) - 1)
    
    return ra_hours, dec_degrees

def parse_file(file_path):
    """
    Parse a .1476 file and return a list of stars (RA, DEC).
    """
    stars = []
    
    with open(file_path, 'rb') as f:
        # Read the header (first 110 bytes)
        header = f.read(110)
        print(header)
        
        record_size = struct.unpack('B', header[-1:])[0]
        
        while True:
            record = f.read(record_size)
            if not record:
                break
            ra, dec = parse_star_record(record)
            stars.append((ra, dec))
    
    return stars


# Example usage
#file_path = 'astap\d20_0101.1476'
stars = []
files = [file for file in os.listdir('astap') if file.startswith("d20")]
for file in files: 
    stars.extend(parse_file('astap/' + file))


#spiral_search_wrapper(stars, search_center_ra=180, search_center_dec=0, search_radius=10, steps=50)
    
    
    #next steps: 
    #1. parse star database and build quads from database 2. match quads 3. output if database and reference image match 















    #hdul[2] is 262144 data point "kdtree_lr". Has one 4-byte, native-endian unsigned int for each leaf node in the tree. For each node, it gives the index of the rightmost data point owned by the node.   
    #hdul[3] is 2557501 pt "kdtree_perm". Contains kd-tree permutation array with unsigned ints for each data point in tree. For each data point, it gives index that the data point had in original array on which the kd tree was built.
    #hdul[4] is 524287 pt "kdtree_bb" contains kdtree bouding box array. Contains 2 3D points for each node in tree. Each data point owned by a node is contained within its bounding box. 
    #hdul[5] is 2557501 pt "kdtree_data" contains the kdtree data, stored as 3D 8-byte native-endian doubles. 
    #print(hdul[5].header)
    #data = hdul[5].data


    #print(data[0])
    #format_string = '<ddd'
    #doubles = struct.unpack(format_string, data[0][0])
    #print("3-d doubles", doubles)

