# Qingbo Liu
# Spring 2020 

import numpy as np

class View:
    def __init__(self):
        self.reset()

    # reset fields to default values 
    def reset(self):
        self.vrp = np.matrix([0.5, 0.5, 1])
        self.vpn = np.matrix([0, 0, -1])
        self.vup = np.matrix([0, 1, 0])
        self.u = np.matrix([-1, 0, 0])
        self.extent = [1.0, 1.0, 1.0]
        self.screen = [400., 400.]
        self.offset = [20., 20.]

    # build a view matrix based on viewing parameters 
    def build(self):
        vtm = np.identity(4, float)

        t1 = np.matrix( [[1, 0, 0, -self.vrp[0, 0]],
                    [0, 1, 0, -self.vrp[0, 1]],
                    [0, 0, 1, -self.vrp[0, 2]],
                    [0, 0, 0, 1] ] )

        vtm = t1 * vtm

        tu = np.cross(self.vup, self.vpn)
        tvup = np.cross(self.vpn, tu)
        tvpn = self.vpn 

        # normalize 
        normalize = lambda x: x / np.linalg.norm(x)
        tu = normalize(tu)
        tvup = normalize(tvup)
        tvpn = normalize(tvpn)

        self.u = tu
        self.vup = tvup
        self.vpn = tvpn

        # align axes 
        r1 = np.matrix( [[ tu[0, 0], tu[0, 1], tu[0, 2], 0.0 ],
                            [ tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0 ],
                            [ tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0 ],
                            [ 0.0, 0.0, 0.0, 1.0 ] ] )
        vtm = r1 * vtm

        # translate view space to the origin 
        T = np.matrix([[1, 0, 0, 0.5*self.extent[0]],
                       [0, 1, 0, 0.5*self.extent[1]],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        vtm = T * vtm

        # scale 
        S = np.matrix([[-self.screen[0]/self.extent[0], 0, 0, 0],
                       [0, -self.screen[1]/self.extent[1], 0, 0],
                       [0, 0, 1.0 / self.extent[2], 0],
                       [0, 0, 0, 1]])
        vtm = S * vtm 

        # add the view offset 
        T = np.matrix([[1, 0, 0, self.screen[0]+self.offset[0]],
                       [0, 1, 0, self.screen[1]+self.offset[1]],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        vtm = T * vtm 

        return vtm

    # returns a new View object with the same set of viewing parameters 
    def clone(self):
        v = View()
         
        v.vrp    = self.vrp 
        v.vpn    = self.vpn 
        v.vup    = self.vup 
        v.u      = self.u 
        v.extent = self.extent 
        v.screen = self.screen
        v.offset = self.offset

        return v 
        

if __name__ == "__main__":
    v = View()
    print(v.build())






