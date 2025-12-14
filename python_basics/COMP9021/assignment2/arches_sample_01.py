from collections import defaultdict

class OpenMeanderError(Exception):
    pass


class OpenMeander:
    def __init__(self, *args):
        
        # pass
        if len(args) < 2 or len(set(args)) != len(args):
            # meiyou print print()
            # 此处是单引号
            raise OpenMeanderError('Not a permutation of {1, ..., n} for some n ≥ 2')
        for n in args:
            if not 1 <= n <= len(args):
                raise OpenMeanderError('Not a permutation of {1, ..., n} for some n ≥ 2')
        
        # 开始尝试构建, 方便下面调用
        self.numbers = args
        self.extended_dyck_word_for_upper_arches = ""
        self.extended_dyck_word_for_lower_arches = ""
        self.points = []
        self.end_points = []
        # 不要这样写
        try:
            self.form_open_meander()
        except Exception as e:
            print(e)

    
    def form_open_meander(self):
        # 默认在下方 -1, 在下，1 在上方
        sign = -1

        x = self.numbers[0]

        lines = {}
        lines[(x, sign)] = '1'
        self.end_points.append((x, sign))

        valid_lines = []

        for i in range(1, len(self.numbers)):
            a, b = self.numbers[i - 1], self.numbers[i]
            # An arch is drawn from left to right if $a_i < a_{i+1}$.
            sign = - sign

            x1, x2 = min(a, b), max(a, b)
            y = abs(b - a) * sign

            # TODO 需要检查是否相交
                
            valid_lines.append((x1, x2, y))
            # check inter section
            
            if a < b:
                lines[(a, y)] = "("
                lines[(b, y)] = ")"
                self.points.append((a, b, y, "("))
            # An arch is drawn from right to left if $a_i > a_{i+1}$.
            else:
                lines[(a, y)] = ")"
                lines[(b, y)] = "("
                self.points.append((a, b, y, ')')) 
            

        x = self.numbers[-1]
        lines[(x,  - sign)] = '1'
        self.end_points.append((x, - sign))

        for x, y in sorted(lines):
            if y > 0:
                self.extended_dyck_word_for_upper_arches += lines[(x, y)]
            else:
                self.extended_dyck_word_for_lower_arches += lines[(x, y)]


    def draw(self, filename, scale=1.0):
         
         with open(filename, 'w') as file:
            # 
            # file.write()
            # file.writelines()
            print("\\documentclass[10pt]{article}", file = file)
            print("\\usepackage{tikz}", file = file)
            print("\\usepackage[margin=0cm]{geometry}", file = file)
            print("\\pagestyle{empty}", file = file)
            print("", file = file)
            print("\\begin{document}", file = file)

            print("", file = file)
            print("\\vspace*{\\fill}", file = file)
            print("\\begin{center}", file = file)

            print("\\begin{tikzpicture}" + f"[x={scale:.1f}cm, y={scale:.1f}cm, very thick]", file = file)
            print(f"\\draw (0,0) -- ({len(self.numbers) + 1},0);", file = file)
            x, y = self.end_points[0]
            print(f"\\draw ({x},0) -- ({x},{ y * scale / 2:.1f});", file = file)

            # TODO 输出 tex 文件

            x, y  = self.end_points[-1]
            print(f"\\draw ({x},0) -- ({x},{ y * scale / 2:.1f});", file = file)

            print("\\end{tikzpicture}", file = file)
            print("\\end{center}", file = file)

            print("\\vspace*{\\fill}", file = file)
            print("", file = file)
            print("\\end{document}", file = file)
        

if __name__ == "__main__":

    try:
        OpenMeander(1, 3, 2, 4)
    except Exception as e:
        print(e)

    m =  OpenMeander(2, 3, 1, 4)
    print(m.extended_dyck_word_for_upper_arches)
    # '(())'
    print(m.extended_dyck_word_for_lower_arches)
    # '(1)1'
    m.draw('open_meanders_1.tex')
    m = OpenMeander(1, 10, 9, 4, 3, 2, 5, 8, 7, 6)
    print(m.extended_dyck_word_for_upper_arches)
    # '(()((())))'
    print(m.extended_dyck_word_for_lower_arches)
    # '1(())1()()'
    m.draw('open_meanders_2.tex', 1.2)
    m = OpenMeander(5, 4, 3, 2, 6, 1, 7, 8, 13, 9, 10, 11, 12)
    m.extended_dyck_word_for_upper_arches
    # '(()())()(()1)'
    print(m.extended_dyck_word_for_lower_arches)
    # '((()1))(()())'
    m.draw('open_meanders_3.tex', 0.7)