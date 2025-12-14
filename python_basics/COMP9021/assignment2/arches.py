from collections import defaultdict

class OpenMeanderError(Exception):
    pass

class DyckWordError(Exception):
    pass

class OpenMeander:
    def __init__(self, *n):

        # Need n >= 2 and distinct integers
        if len(n) < 2:
            raise OpenMeanderError("Not a permutation of {1, ..., n} for some n ≥ 2")

        if len(n) != len(set(n)):
            raise OpenMeanderError("Not a permutation of {1, ..., n} for some n ≥ 2")
        
        # Each number should in the range of 1 to n
        for number in n:
            if number < 1 or number > len(n):
                raise OpenMeanderError("Not a permutation of {1, ..., n} for some n ≥ 2")
            
        self.numbers = n
        self.extended_dyck_word_for_upper_arches = ""
        self.extended_dyck_word_for_lower_arches = ""
        self.start_point = tuple()
        self.end_point = tuple()
        self.latex = []

        self.create_meander()
        
    def create_meander(self):
        # An arc can be defined as start point end point and vertex
        # Think arc is a half of circle, then diameter is abs(x2 - x1)
        # And vertex is ((x2 - x1)/2, (x2 - x1)/2)

        direction = -1 # -1 means lower, 1 means upper
        dyck_words = {}

        arcs = []

        # Starting point from lower and record as 1
        start_point = self.numbers[0]
        dyck_words[(start_point, direction)] = "1"
        self.start_point = (start_point, direction)

        for i in range(1, len(self.numbers)):
            a1 = self.numbers[i-1]
            a2 = self.numbers[i]

            # The arc is draw in upper and lower cycle
            direction = (-1) * direction

            left_point = min(a1, a2)
            right_point = max(a1, a2)
            radius = (right_point - left_point) / 2
            y_vertex = (right_point - left_point) / 2 * direction

            is_intersect = self.check_arc_intersect(arcs, left_point, right_point, direction)

            if is_intersect:
                raise OpenMeanderError("Does not define an open meander")
            else:
                arcs.append((left_point, right_point, direction))
            
            # Left side of the arc is (, and left side of the arc is )
            dyck_words[(left_point, y_vertex)] = "("
            dyck_words[(right_point, y_vertex)] = ")"

            # Collect drawing direction for latex file
            if a1 < a2:
                # Draw from left to right, define as 1
                self.latex.append((a1, a2, radius, direction, 1))
            if a1 > a2:
                # Draw from right to left, define as -1
                self.latex.append((a1, a2, radius, direction, -1))

        end_point = self.numbers[-1]
        dyck_words[(end_point, (-1)*direction)] = "1"
        self.end_point = (end_point, (-1)*direction)

        # Add dyck words into 2 attributes based on instruction
        for x, y in sorted(dyck_words):
            if y < 0:
                self.extended_dyck_word_for_lower_arches += dyck_words[(x, y)]
            else:
                self.extended_dyck_word_for_upper_arches += dyck_words[(x, y)]
            
    def check_arc_intersect(self, arcs, left_point, right_point, direction):
        for arc in arcs:
            # If in the same side
            if arc[2] == direction:
                # To check intersection is to see if one point of the new arc is within previous arcs
                # and the other point is out of previsou arcs
                if arc[0] < left_point < arc[1] and right_point > arc[1]:
                    return True
                
                if arc[0] < right_point < arc[1] and left_point < arc[0]:
                    return True
                
            else:
                continue

        return False
    
    def draw(self, filename, scale=1):
        with open(filename, 'w') as file:
            file.write(
                "\\documentclass[10pt]{article}\n"
                "\\usepackage{tikz}\n"
                "\\usepackage[margin=0cm]{geometry}\n"
                "\\pagestyle{empty}\n"
                "\n"
                "\\begin{document}\n"
                "\n"
                "\\vspace*{\\fill}\n"
                "\\begin{center}\n"
                f"\\begin{{tikzpicture}}[x={scale:.1f}cm, y={scale:.1f}cm, very thick]\n"
                f"\\draw (0,0) -- ({len(self.numbers) + 1},0);\n"
                f"\\draw ({self.start_point[0]},0) -- ({self.start_point[0]},{self.start_point[1]*scale/2:.1f});\n"
            )

            for arc in self.latex:
                start_angle = arc[3] * arc[4] * 180
                radius = arc[4] * arc[2]
                file.write(f"\\draw ({arc[0]},0) arc[start angle={start_angle}, end angle=0, radius={radius}];\n")
            
            file.write(
                f"\\draw ({self.end_point[0]},0) -- ({self.end_point[0]},{self.end_point[1]*scale/2:.1f});\n"
                "\\end{tikzpicture}\n"
                "\\end{center}\n"
                "\\vspace*{\\fill}\n"
                "\n"
                "\\end{document}"
                "\n"
            )


class DyckWord:
    def __init__(self, s):
        # If s is empty string
        if not s:
            raise DyckWordError("Expression should not be empty")

        # If s only contains ( and )
        new_s = s.replace("(", "").replace(")", "")
        if new_s:
            raise DyckWordError("Expression can only contain '(' and ')'")
        
        # Implement bracket matching algorithm learnt from 9024
        # If balanced, we can get parenthesis pairs 
        self.is_balance = True
        self.pairs = [] # We find in the list of pairs, those archs with the lower depth occur at first
        self.bracket_matching(s)

        if not self.is_balance:
            raise DyckWordError("Unbalanced parentheses in expression")

        self.dyck_word = s
        self.depth_report = defaultdict(int)
        # print(self.pairs)
        
        self.get_depths()
        # print(self.depth_report)

    def bracket_matching(self, s):
        stack = []
        pairs = []

        for i , char in enumerate(s):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if not stack:
                    self.is_balance = False
                    return

                left_index = stack.pop()
                right_index = i
                pairs.append((left_index, right_index))
        
        if stack:
            self.is_balance = False
            return
        else:
            self.pairs = pairs

    def get_depths(self):
        for left_arc, right_arc in self.pairs:
            # Arc with depth 0
            if right_arc - left_arc == 1:
                self.depth_report[(left_arc, right_arc)] = 0
            else:
                inner_depths = []
                for inner_left, inner_right in self.depth_report:
                    # Record inner arcs' depths
                    if inner_left > left_arc and inner_right < right_arc:
                        inner_depths.append(self.depth_report[inner_left, inner_right])

                # Outer arc's depth is max of inner depths plus 1
                self.depth_report[(left_arc, right_arc)] = max(inner_depths) + 1

    def report_on_depths(self):
        # Print result
        depth_result = {}
        for depth in self.depth_report.values():
            if depth not in depth_result:
                depth_result[depth] = 1
            else:
                depth_result[depth] += 1

        for depth in sorted(depth_result):
            if depth_result[depth] == 1:
                print(f"There is {depth_result[depth]} arch of depth {depth}.")        
            if depth_result[depth] > 1:
                print(f"There are {depth_result[depth]} arches of depth {depth}.")

    def find_path(self, visited, height, start_point, end_point, path):
        x = start_point[0]
        y = start_point[1]
        # If point is visited
        if start_point in visited:
            return
        
        # If point reaches end point
        if start_point == end_point:
            # Put end point into path
            path.append(end_point)
            return
        
        # Path cannot exceed range
        if y > height or y < 0 or x > end_point[0]:
            return
        
        path.append(start_point)
        visited.append(start_point)
        # First check right side, if there is a point, go up because we draw from left to right
        if (x+1, y) in visited:
            new_point = (x, y+1)
            self.find_path(visited, height, new_point, end_point, path)

        # If there is no point in the right, and there is a point below, then path go right
        if (x+1, y) not in visited and (x, y-1) in visited:
            new_point = (x+1, y)
            self.find_path(visited, height, new_point, end_point, path)

        # If there is no point below , then go down
        if (x, y-1) not in visited:
            new_point = (x, y-1)
            self.find_path(visited, height, new_point, end_point, path)

    def draw_arches(self, filename, scale=1):
        with open(filename, 'w') as file:
            file.write(
                "\\documentclass[10pt]{article}\n"
                "\\usepackage{tikz}\n"
                "\\usepackage[margin=0cm]{geometry}\n"
                "\\pagestyle{empty}\n"
                "\n"
                "\\begin{document}\n"
                "\n"
                "\\vspace*{\\fill}\n"
                "\\begin{center}\n"
                f"\\begin{{tikzpicture}}[x={scale:.1f}cm, y={scale:.1f}cm, very thick]\n"
                f"\\draw (-1,0) -- ({len(self.dyck_word)},0);\n"
            )

            arc_path = {} # This is for color dyck word, list of pathes keyed by depth
            arc_path_list = [] # This is for draw dyck word, a list of every path
            visited = []
            for pair, depth in self.depth_report.items():
                # Find arc from small to big depth, record arc's vertices as visited
                if depth == 0:
                    vertex1 = (pair[0], 0)
                    vertex2 = (pair[0], 1)
                    vertex3 = (pair[1], 1)
                    vertex4 = (pair[1], 0)
                    visited.append(vertex1)
                    visited.append(vertex2)
                    visited.append(vertex3)
                    visited.append(vertex4)
                    
                    path = [vertex1, vertex2, vertex3, vertex4]
                    arc_path_list.append(path)
                    if depth not in arc_path:
                        arc_path[depth] = [path]
                    else:
                        arc_path[depth].append(path)

                # Find a path from start point to end point, max height is depth + 1
                path = []
                height = depth + 1
                self.find_path(visited, height, (pair[0], 0), (pair[1], 0), path)

                if path:
                    # But right now the path includs all point for this arc
                    # to draw the arc, we only need verties
                    start = path[0]
                    end = path[-1]
                    new_path = []
                    new_path.append(start)
                    x_y_change = []
                    # We can check x or y changes to find a direction turn for the arc
                    for i in range(1, len(path)):
                        first_x = path[i - 1][0]
                        first_y = path[i - 1][1]
                        second_x = path[i][0]
                        second_y = path[i][1]
                        # Record x y changes for every point
                        x_y_change.append(((second_x - first_x), (second_y - first_y)))

                    for i in range(1, len(x_y_change)):
                        if x_y_change[i] != x_y_change[i - 1]:
                            # If x y changes dont match, it means this is a turn
                            new_path.append(path[i])

                    new_path.append(end)
                    arc_path_list.append(new_path)
                    if new_path:
                        if depth not in arc_path:
                            arc_path[depth] = [new_path]
                        else:
                            arc_path[depth].append(new_path)

            # print(arc_path_list)
            # print(arc_path)
            # Draw from outer arc to inner arc
            for idx, path in enumerate(sorted(arc_path_list)):
                line = " -- ".join(f"({x},{y})" for x, y in path)
                file.write(
                    f"% Arch {idx + 1}\n"
                    f"    \\draw {line};\n"
                )

            file.write(
                "\\end{tikzpicture}\n"
                "\\end{center}\n"
                "\\vspace*{\\fill}\n"
                "\n"
                "\\end{document}"
                "\n"
            )

    def colour_arches(self, filename, scale=1):
        with open(filename, 'w') as file:
            file.write(
                "\\documentclass[10pt]{article}\n"
                "\\usepackage[dvipsnames]{xcolor}\n"
                "\\usepackage{tikz}\n"
                "\\usepackage[margin=0cm]{geometry}\n"
                "\\pagestyle{empty}\n"
                "\n"
                "\\begin{document}\n"
                "\n"
                "\\vspace*{\\fill}\n"
                "\\begin{center}\n"
                f"\\begin{{tikzpicture}}[x={scale:.1f}cm, y={scale:.1f}cm, very thick]\n"
                f"\\draw (-1,0) -- ({len(self.dyck_word)},0);\n"
            )

            arc_path = {} # This is for color dyck word, list of pathes keyed by depth
            arc_path_list = [] # This is for draw dyck word, a list of every path
            visited = []
            for pair, depth in self.depth_report.items():
                # Find arc from small to big depth, record arc's vertices as visited
                if depth == 0:
                    vertex1 = (pair[0], 0)
                    vertex2 = (pair[0], 1)
                    vertex3 = (pair[1], 1)
                    vertex4 = (pair[1], 0)
                    visited.append(vertex1)
                    visited.append(vertex2)
                    visited.append(vertex3)
                    visited.append(vertex4)
                    
                    path = [vertex1, vertex2, vertex3, vertex4]
                    arc_path_list.append(path)
                    if depth not in arc_path:
                        arc_path[depth] = [path]
                    else:
                        arc_path[depth].append(path)

                # Find a path from start point to end point, max height is depth + 1
                path = []
                height = depth + 1
                self.find_path(visited, height, (pair[0], 0), (pair[1], 0), path)

                if path:
                    # But right now the path includs all point for this arc
                    # to draw the arc, we only need verties
                    start = path[0]
                    end = path[-1]
                    new_path = []
                    new_path.append(start)
                    x_y_change = []
                    # We can check x or y changes to find a direction turn for the arc
                    for i in range(1, len(path)):
                        first_x = path[i - 1][0]
                        first_y = path[i - 1][1]
                        second_x = path[i][0]
                        second_y = path[i][1]
                        # Record x y changes for every point
                        x_y_change.append(((second_x - first_x), (second_y - first_y)))

                    for i in range(1, len(x_y_change)):
                        if x_y_change[i] != x_y_change[i - 1]:
                            # If x y changes dont match, it means this is a turn
                            new_path.append(path[i])

                    new_path.append(end)
                    arc_path_list.append(new_path)
                    if new_path:
                        if depth not in arc_path:
                            arc_path[depth] = [new_path]
                        else:
                            arc_path[depth].append(new_path)

            # print(arc_path_list)
            # print(arc_path)
            colors = {0:"Red", 1:"Orange", 2:"Goldenrod", 3:"Yellow", 4:"LimeGreen", 5:"Green", 6:"Cyan", 7:"SkyBlue", 8:"Blue", 9:"Purple"}
            # Draw from outer arc to inner arc
            # we need to reverse the dict then draw the arc with biggest depth first
            depths = []
            for key in arc_path.keys():
                depths.append(key)
            
            depths = depths[::-1]

            for depth in depths:
                file.write(f"% Arches of depth {depth}\n")
                color = colors[depth % 10]
                for path in sorted(arc_path[depth]):
                    line = " -- ".join(f"({x},{y})" for x, y in path)
                    file.write(f"    \\draw[fill={color}] {line};\n")

            file.write(
                "\\end{tikzpicture}\n"
                "\\end{center}\n"
                "\\vspace*{\\fill}\n"
                "\n"
                "\\end{document}"
                "\n"
            )

# if __name__ == "__main__":
    # try:
    #     OpenMeander(1, 3, 2, 4)
    # except Exception as e:
    #     print(e)

    # m =  OpenMeander(2, 3, 1, 4)
    # print(m.extended_dyck_word_for_upper_arches)
    # # '(())'
    # print(m.extended_dyck_word_for_lower_arches)
    # # '(1)1'
    # m.draw('open_meanders_1.tex')
    # m = OpenMeander(1, 10, 9, 4, 3, 2, 5, 8, 7, 6)
    # print(m.extended_dyck_word_for_upper_arches)
    # # '(()((())))'
    # print(m.extended_dyck_word_for_lower_arches)
    # # '1(())1()()'
    # m.draw('open_meanders_2.tex', 1.2)
    # m = OpenMeander(5, 4, 3, 2, 6, 1, 7, 8, 13, 9, 10, 11, 12)
    # print(m.extended_dyck_word_for_upper_arches)
    # # '(()())()(()1)'
    # print(m.extended_dyck_word_for_lower_arches)
    # # '((()1))(()())'
    # m.draw('open_meanders_3.tex', 0.7)

    # m = DyckWord("(((((((((((((())))))))))))))")
    # m.draw_arches('draw_arc_1.tex')
    # m.colour_arches('color_arc_1.tex')
    # m = DyckWord("(()(()(())))")
    # m.draw_arches('draw_arc_2.tex')
    # m.colour_arches('color_arc_2.tex')
    # DyckWord("((()())(()(()())))")
    # m = DyckWord("((()(()())(()(()(())))((()()))()(()())))")
    # m.draw_arches('draw_arc_4.tex', 0.3)
    # m.colour_arches('color_arc_4.tex', 0.3)