# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared geometry definitions for independent unconstrained ethylene audits."""
import math


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def unit(a):
    length=math.sqrt(dot(a,a))
    if length<1e-12:
        raise ValueError('Undefined geometric angle at a collinear/zero-length vector')
    return [x/length for x in a]


def dihedral(positions,a,b,c,d):
    axis=unit(sub(positions[c],positions[b]))
    left,right=sub(positions[a],positions[b]),sub(positions[d],positions[c])
    left=sub(left,[dot(left,axis)*x for x in axis])
    right=sub(right,[dot(right,axis)*x for x in axis])
    unit(left);unit(right)
    return math.degrees(math.atan2(dot(cross(axis,left),right),dot(left,right)))


def geometry(run):
    positions=run['accepted_frames'][-1]['positions_angstrom']
    neighbors=[[h for h in range(2,6)
                if dot(sub(positions[h],positions[c]),sub(positions[h],positions[c]))
                <dot(sub(positions[h],positions[1-c]),sub(positions[h],positions[1-c]))]
               for c in (0,1)]
    if run['numbers']!=[6,6,1,1,1,1] or neighbors!=[[2,3],[4,5]]:
        raise ValueError('The documented C0/C1/H2/H3/H4/H5 atom ordering has changed')
    normals=[unit(cross(sub(positions[h[0]],positions[c]),sub(positions[h[1]],positions[c])))
             for c,h in enumerate(neighbors)]
    return dict(ch2_plane_acute_angle_degrees=math.degrees(math.acos(min(1.,abs(dot(*normals))))),
        dihedrals_degrees={f'H{a}-C0-C1-H{b}':dihedral(positions,a,0,1,b)
                          for a in neighbors[0] for b in neighbors[1]},
        definition='signed H-C-C-H torsions in [-180,180]; acute angle between C0-H2-H3 and C1-H4-H5 planes',
        neighbors=neighbors)
