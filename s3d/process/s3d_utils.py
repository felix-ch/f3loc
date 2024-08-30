'''
BORROWED FROM LASER https://github.com/zillow/laser
'''

import numpy as np

def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices"""
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def poly_verts_to_lines_append_head(verts):
    n_verts = verts.shape[0]
    if n_verts == 0:
        return
    assert n_verts > 1
    verts = np.concatenate([verts, verts[0:1]], axis=0)  # append head to tail
    lines = np.stack([verts[:-1], verts[1:]], axis=1)  # N,2,2
    return lines


def read_s3d_floorplan(annos):
    """visualize floorplan"""
    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos["semantics"]:
        for planeID in semantic["planeID"]:
            if annos["planes"][planeID]["type"] == "floor":
                planes.append({"planeID": planeID, "type": semantic["type"]})

        if semantic["type"] == "outwall":
            outerwall_planes = semantic["planeID"]

    # extract hole vertices
    lines_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                lines_holes.extend(
                    np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
                )
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc["coordinate"] for junc in annos["junctions"]])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][plane["planeID"]]))[
            0
        ].tolist()
        junction_pairs = [
            np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane["type"]])
    """
    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.append([outerwall_polygon[0], 'outwall'])
    """

    junctions = np.array([junc["coordinate"][:2] for junc in annos["junctions"]])
    door_lines = []
    window_lines = []
    room_lines = []
    n_rooms = 0
    for (polygon, poly_type) in polygons:
        polygon = junctions[np.array(polygon)] / 1000.0  # mm to meter
        lines = poly_verts_to_lines_append_head(polygon)
        if not is_polygon_clockwise(lines):
            lines = poly_verts_to_lines_append_head(np.flip(polygon, axis=0))
        if poly_type == "door":
            door_lines.append(lines)
        elif poly_type == "window":
            window_lines.append(lines)
        else:
            n_rooms += 1
            room_lines.append(lines)

    room_lines = np.concatenate(room_lines, axis=0)
    door_lines = (
        np.zeros((0, 2, 2), float)
        if len(door_lines) == 0
        else np.concatenate(door_lines, axis=0)
    )
    window_lines = (
        np.zeros((0, 2, 2), float)
        if len(window_lines) == 0
        else np.concatenate(window_lines, axis=0)
    )

    return n_rooms, room_lines, door_lines, window_lines

def is_polygon_clockwise(lines):
    return np.sum(np.cross(lines[:, 0], lines[:, 1], axis=-1)) > 0

