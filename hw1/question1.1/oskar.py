import sys
from collections import defaultdict

FREE = "0"


def parse_matrix_from_str_lines(lines):
    """
    Parses and returns matrix from passed string lines.
    """
    mat = [row.split() for row in lines]
    return mat


def is_valid_position(position, xy_mat, yz_mat, zx_mat):
    """
    Returns True if there is no obstacle in position, False otherwise.
    """
    x, y, z = position
    if (xy_mat[y][x] == FREE) and (yz_mat[z][y] == FREE) and (zx_mat[x][z] == FREE):
        return True
    return False


def get_possible_positions(x, y, z, xy_mat, yz_mat, zx_mat, visited_dict):
    """
    Returns all possible locations around position that are valid and haven't been visited yet,
    together with the command that created them from the original.
    """
    dim_x = len(zx_mat)
    dim_y = len(xy_mat)
    dim_z = len(yz_mat)
    positions = []
    if x < dim_x - 1:
        pos = (x + 1, y, z)
        command = 0
        if (
            is_valid_position(position=pos, xy_mat=xy_mat, yz_mat=yz_mat, zx_mat=zx_mat)
            and visited_dict[pos] == False
        ):
            positions.append((pos, command))
    if x > 0:
        pos = (x - 1, y, z)
        command = 1
        if (
            is_valid_position(position=pos, xy_mat=xy_mat, yz_mat=yz_mat, zx_mat=zx_mat)
            and visited_dict[pos] == False
        ):
            positions.append((pos, command))

    if y < dim_y - 1:
        pos = (x, y + 1, z)
        command = 2
        if (
            is_valid_position(position=pos, xy_mat=xy_mat, yz_mat=yz_mat, zx_mat=zx_mat)
            and visited_dict[pos] == False
        ):
            positions.append((pos, command))
    if y > 0:
        pos = (x, y - 1, z)
        command = 3
        if (
            is_valid_position(position=pos, xy_mat=xy_mat, yz_mat=yz_mat, zx_mat=zx_mat)
            and visited_dict[pos] == False
        ):
            positions.append((pos, command))

    if z < dim_z - 1:
        pos = (x, y, z + 1)
        command = 4
        if (
            is_valid_position(position=pos, xy_mat=xy_mat, yz_mat=yz_mat, zx_mat=zx_mat)
            and visited_dict[pos] == False
        ):
            positions.append((pos, command))
    if z > 0:
        pos = (x, y, z - 1)
        command = 5
        if (
            is_valid_position(position=pos, xy_mat=xy_mat, yz_mat=yz_mat, zx_mat=zx_mat)
            and visited_dict[pos] == False
        ):
            positions.append((pos, command))
    return positions


def find_path_for_start_and_destination(sx, sy, sz, dx, dy, dz, xy_mat, yz_mat, zx_mat):
    """
    Returns path between start and destination passed as coords, using the obstacle matrices.
    """
    visited_dict = defaultdict(lambda: False)
    # queue of pairs: (tuple of position, list of commands took from start)
    positions_queue = [((sx, sy, sz), [])]
    while len(positions_queue) != 0:
        position_tuple, curr_path = positions_queue.pop(0)
        if visited_dict[position_tuple] == True:
            continue
        visited_dict[position_tuple] = True
        pos_x, pos_y, pos_z = position_tuple
        new_positions = get_possible_positions(
            x=pos_x,
            y=pos_y,
            z=pos_z,
            xy_mat=xy_mat,
            yz_mat=yz_mat,
            zx_mat=zx_mat,
            visited_dict=visited_dict,
        )
        for new_pos, command in new_positions:
            new_path = list(curr_path)
            new_path.append(command)
            if new_pos[0] == dx and new_pos[1] == dy and new_pos[2] == dz:
                return new_path  # got to destination
            positions_queue.append((new_pos, new_path))
    return None


def main():
    NUM_ARGS = 7
    args = sys.argv
    assert (
        len(args) == NUM_ARGS + 1
    ), "Passed wrong number of arguments, expected {NUM_ARGS=}"

    # First argument is the name of the code file.
    sx_str, sy_str, sz_str, dx_str, dy_str, dz_str, filename = sys.argv[1:]
    sx = int(sx_str)
    sy = int(sy_str)
    sz = int(sz_str)
    dx = int(dx_str)
    dy = int(dy_str)
    dz = int(dz_str)
    # print(f"{sx=},{sy=},{sz=},{dx=},{dy=},{dz=},{filename=}") # TODO DELETE
    file_reader = open(filename, "r")
    file_content = file_reader.readlines()
    file_reader.close()
    curr_line = 0
    [x_dim, y_dim, z_dim] = [int(val) for val in file_content[curr_line].split()]
    curr_line += 1

    xy_mat = parse_matrix_from_str_lines(
        lines=file_content[curr_line : (curr_line + y_dim)]
    )
    curr_line += y_dim + 1  # blank line after matrix

    yz_mat = parse_matrix_from_str_lines(
        lines=file_content[curr_line : (curr_line + z_dim)]
    )
    curr_line += z_dim + 1  # blank line after matrix

    zx_mat = parse_matrix_from_str_lines(
        lines=file_content[curr_line : (curr_line + x_dim)]
    )

    assert is_valid_position(
        (sx, sy, sz), xy_mat, yz_mat, zx_mat
    ), "Passed start position is invalid"

    assert is_valid_position(
        (dx, dy, dz), xy_mat, yz_mat, zx_mat
    ), "Passed end position is invalid"

    path = find_path_for_start_and_destination(
        sx, sy, sz, dx, dy, dz, xy_mat, yz_mat, zx_mat
    )

    if path is None:
        print("Found no path")
    else:
        print(f"{sx} {sy} {sz}")
        print(f"{dx} {dy} {dz}")
        path_str = " ".join(str(command) for command in path)
        print(path_str)


main()
