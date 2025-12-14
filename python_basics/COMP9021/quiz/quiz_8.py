# Written by *** for COMP9021


# Write a program that models points, lines, and parallelograms
# in a 2D plane using object-oriented programming.
# Your implementation should:
# - Validate geometric consistency,
# - Represent shapes in human-readable form, and
# - Detect whether an additional line divides a parallelogram into two smaller parallelograms.
#
# The program does not require direct user input for this exercise.
# Instead, you will create and test objects directly in code.
# The following classes must be implemented:
# - Point(x, y)
#   Represents a point with integer coordinates.
# - Line(pt_1, pt_2)
#   Represents the infinite line passing through two distinct Point objects.
# - Parallelogram(line_1, line_2, line_3, line_4)
#   Represents a parallelogram defined by four Line objects.
#
# Validation Rules
# - Point:
#   Both arguments must be integers.
#   If not, raise a PointError:
#       Cannot create a Point: expected integer coordinates, got (...)
# - Line: Both arguments must be Point objects.
#   The two points must be distinct.
#   If either rule fails, raise a LineError:
#     Cannot create a Line: both arguments must be Point objects, got (...)
#   or
#     Cannot create a Line: both points are identical (A point with x = ... and y = ...)
# - Parallelogram:
#   A parallelogram must be created from exactly four Line objects.
#   Each of the four arguments must be a valid instance of Line,
#   and together they must define two pairs of distinct, parallel lines,
#   with at least one of these pairs being horizontal (slope 0) or vertical (slope ∞).
#   If either rule is violated, raise a ParallelogramError
#   with one of the following messages:
#   - If the arguments are invalid or the wrong number of lines are provided:
#         Cannot create a Parallelogram: expected ... Line objects, got 3 object(s) of type(s) (...)
#   - If the lines do not define a valid parallelogram
#     (for example, they are not parallel in pairs or coincide):
#         Cannot create a Parallelogram: lines do not define a parallelogram
#
# Each class must define both __repr__ and __str__ methods for user-friendly output.
# See the sample outputs.
#
# Within Parallelogram, implement the method
# divides_into_two_parallelograms(self, line)
# A line divides a parallelogram into two smaller parallelograms if:
# - It is parallel to one pair of opposite sides, and
# - Its intercept lies strictly between those of the two sides.
# Return True if both conditions are met; otherwise, return False.
#
# Notes:
# - You can assume that Parallelogram() will always be passed
#       at least 2 arguments.
# - You will likely need to use the type() function
#   together with the __name__ attribute of class objects.



# INSERT YOUR CODE HERE
class PointError(Exception):
    pass

class LineError(Exception):
    pass

class ParallelogramError(Exception):
    pass

class Point:
    def __init__(self, x, y):
        if isinstance(x, int) and isinstance(y, int):
            self.x = x
            self.y = y
        else:
            type_x = type(x).__name__
            type_y = type(y).__name__

            raise PointError(f"Cannot create a Point: expected integer coordinates, got ({type_x}, {type_y})")  

    def __str__(self):
        return f"A point with x = {self.x} and y = {self.y}"
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
        
    
class Line:
    def __init__(self, pt_1, pt_2):
        if not isinstance(pt_1, Point) or not isinstance(pt_2, Point):
            type_pt_1 = type(pt_1).__name__
            type_pt_2 = type(pt_2).__name__
            raise LineError(f"Cannot create a Line: both arguments must be Point objects, got ({type_pt_1}, {type_pt_2})")
        
        if pt_1.x == pt_2.x and pt_1.y == pt_2.y:
            raise LineError(f"Cannot create a Line: both points are identical (A point with x = {pt_1.x} and y = {pt_1.y})")
        
        # For a infinite line passing two distince points, we can calculate its straight line equation y = ax + b
        self.pt_1 = pt_1
        self.pt_2 = pt_2

        # If it is vertical line, slop = infintie
        if pt_1.x == pt_2.x:
            self.is_vertical = True
            self.is_horizontal = False
            self.a = None
            self.b = None
        else:
            # Otherwise, slope can be calculated
            self.is_vertical = False
            self.a = (pt_2.y - pt_1.y) / (pt_2.x - pt_1.x)
            if self.a == 0:
                self.is_horizontal = True
            else:
                self.is_horizontal = False
            self.b = self.get_y_intercept(self.a, pt_1.x, pt_1.y)

    def __str__(self):
        return f"A line passing through:\n  - A point with x = {self.pt_1.x} and y = {self.pt_1.y}\n  - A point with x = {self.pt_2.x} and y = {self.pt_2.y}"
    
    def __repr__(self):
        return f"Line(Point({self.pt_1.x}, {self.pt_1.y}), Point({self.pt_2.x}, {self.pt_2.y}))"

    def get_y_intercept(self, slope, x, y):
        intercept = y - slope * x
        return intercept

    def check_parallel(self, other_line):
        # If two lines have the same slope, they are parallel. Or they are both vertical
        if self.is_vertical and other_line.is_vertical:
            return True
        
        if not self.is_vertical and not other_line.is_vertical:
            if self.a == other_line.a:
                return True
            
        return False
        
    def __eq__(self, other_line):
        slope1 = self.a
        y_inter1 = self.b
        slope2 = other_line.a
        y_inter2 = other_line.b

        # If both are vertical and check their x
        if self.is_vertical and other_line.is_vertical:
            if self.pt_1.x == other_line.pt_1.x:
                return True
            else:
                return False
  
        # Check slope and y intercept   
        if slope1 == slope2 and y_inter1 == y_inter2:
            return True
        
        return False

    def get_intercection(self, other_line):
        pass

class Parallelogram:
    def __init__(self, *lines):
        types = []
        for line in lines:
            types.append(type(line).__name__)

        # Check 4 lines types
        is_line = True
        for name in types:
            if name != "Line":
                is_line = False
        
        if len(types) != 4 or not is_line:
            error_message = f"Cannot create a Parallelogram: expected 4 Line objects, got {len(types)} object(s) of type(s) ("
            
            if types:
                error_message += f"{', '.join(types)})"
            
            raise ParallelogramError(error_message)

        self.lines = list(lines)

        # Check coincide
        for i in range(4):
            for j in range(i+1, 4):
                if self.lines[i] == self.lines[j]:
                    raise ParallelogramError("Cannot create a Parallelogram: lines do not define a parallelogram")
        
        # Check parallel??
        

    def __str__(self):
        text = "A parallelogram built from:"

        for line in self.lines:
            text += "\n  * A line passing through:\n"
            text += f"    - A point with x = {line.pt_1.x} and y = {line.pt_1.y}\n"
            text += f"    - A point with x = {line.pt_2.x} and y = {line.pt_2.y}"

        return text
    
    def __repr__(self):
        text = "Parallelogram("

        for i, line in enumerate(self.lines):
            line_text = repr(line)

            if i == 0:
                text += line_text + ","
            else:
                text += "\n" + " " * 14 + line_text + ","

        text = text.rstrip(",")
        text += "\n" + " " * 13 + ")"

        return text
    
# if __name__ == "__main__":

#     p1 = Point(0, 0)
#     p2 = Point(3, 0)
#     p3 = Point(0, 2)
#     p4 = Point(3, 2)
#     p5 = Point(3, 1)
#     p6 = Point(3, 3)

#     line1 = Line(p1, p5)
#     line2 = Line(p1, p3)
#     line3 = Line(p2, p4)
#     line4 = Line(p3, p6)

#     para2 = Parallelogram(line1, line2, line4, line3)

#     # print(para2.divides_into_two_parallelograms(line3))