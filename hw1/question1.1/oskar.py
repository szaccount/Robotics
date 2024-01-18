import sys


def parse_matrix_from_str_lines(lines):
    """
    Parses and returns matrix from passed string lines.
    """
    mat = [row.split() for row in lines]
    return mat


def main():
    NUM_ARGS = 7
    args = sys.argv
    assert (
        len(args) == NUM_ARGS + 1
    ), "Passed wrong number of arguments, expected {NUM_ARGS=}"

    sx, sy, sz, dx, dy, dz, filename = sys.argv[1:]
    print(f"{sx=},{sy=},{sz=},{dx=},{dy=},{dz=},{filename=}")
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

    

main()
