# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# Returns a standard filename based on input number.

def filestr(imgnum):
    if imgnum < 10:
        imgstr = "0000%s.jpg" % (imgnum)
    elif imgnum < 100:
        imgstr = "000%s.jpg" % (imgnum)
    elif imgnum < 1000:
        imgstr = "00%s.jpg" % (imgnum)
    elif imgnum < 10000:
        imgstr = "0%s.jpg" % (imgnum)
    elif imgnum < 100000:
        imgstr = "%s.jpg" % (imgnum)

    return imgstr