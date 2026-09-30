"""Bambu Studio project 3MF with per-triangle colour painting (the layout Bambu itself and Meshy use).
Opens with the printer preset and one PLA filament per palette colour already set."""
import json
import uuid

import numpy as np

# printer: (Bambu printer name, preset suffix, bed centre mm)
PRINTERS = {
    'X1C': ("Bambu Lab X1 Carbon", "X1C", 128), 'X1E': ("Bambu Lab X1E", "X1E", 128),
    'P1S': ("Bambu Lab P1S", "P1S", 128), 'P1P': ("Bambu Lab P1P", "P1P", 128),
    'A1': ("Bambu Lab A1", "A1", 128), 'A1M': ("Bambu Lab A1 mini", "A1M", 90),
}


def paint_code(state):
    """Bambu/PrusaSlicer TriangleSelector code for a whole (unsplit) triangle painted with extruder `state`."""
    if state < 3:
        return format(state << 2, 'X')
    n, nibbles = state - 3, [0b1100]
    while n >= 15:
        nibbles.append(0b1111); n -= 15
    nibbles.append(n)
    return ''.join(format(x, 'X') for x in reversed(nibbles))


def triangles(ob, depsgraph, mm):
    """World-space vertices in mm, triangle vertex indices, material index per triangle."""
    ev = ob.evaluated_get(depsgraph)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
    M = np.array(ob.matrix_world)
    co = (co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]) * mm
    nt = len(me.loop_triangles)
    tri = np.empty(nt * 3, dtype=np.int64); me.loop_triangles.foreach_get('vertices', tri)
    mat = np.empty(nt, dtype=np.int64); me.loop_triangles.foreach_get('material_index', mat)
    ev.to_mesh_clear()
    return co, tri.reshape(-1, 3), mat


def package(name, co, tri, extruder, colors, printer):
    """-> list of (path inside the zip, text). extruder: 1-based filament per triangle."""
    model_name, suffix, centre = PRINTERS.get(printer, PRINTERS['X1C'])
    n = len(colors)
    codes = np.array([paint_code(s) for s in range(1, 17)])[np.clip(extruder, 1, 16) - 1]
    u = lambda: str(uuid.uuid4())
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
            'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
            'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n')
    obj = [head, ' <metadata name="BambuStudio:3mfVersion">2</metadata>\n <resources>\n',
           f'  <object id="1" p:UUID="{u()}" type="model">\n   <mesh>\n    <vertices>\n']
    obj += [f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n' for x, y, z in co]
    obj.append('    </vertices>\n    <triangles>\n')
    obj += [f'     <triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{pc}"/>\n' for (a, b, c), pc in zip(tri, codes)]
    obj.append('    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n')
    main = (head + ' <metadata name="Application">BambuStudio-02.03.01.00</metadata>\n'
            ' <metadata name="BambuStudio:3mfVersion">2</metadata>\n'
            ' <metadata name="BambuStudio:MmPaintingVersion">1</metadata>\n'
            f' <metadata name="Title">{name}</metadata>\n <resources>\n'
            f'  <object id="2" p:UUID="{u()}" type="model">\n   <components>\n'
            f'    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{u()}" '
            'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n   </components>\n  </object>\n </resources>\n'
            f' <build p:UUID="{u()}">\n  <item objectid="2" p:UUID="{u()}" '
            f'transform="1 0 0 0 1 0 0 0 1 {centre} {centre} 0" printable="1"/>\n </build>\n</model>\n')
    settings = (f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="2">\n'
                f'    <metadata key="name" value="{name}"/>\n    <metadata key="extruder" value="1"/>\n'
                f'    <metadata face_count="{len(tri)}"/>\n    <part id="1" subtype="normal_part">\n'
                f'      <metadata key="name" value="{name}"/>\n'
                '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                f'      <mesh_stat face_count="{len(tri)}" edges_fixed="0" degenerate_facets="0" facets_removed="0" '
                'facets_reversed="0" backwards_edges="0"/>\n    </part>\n  </object>\n  <plate>\n'
                '    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n'
                '    <metadata key="locked" value="false"/>\n    <model_instance>\n'
                '      <metadata key="object_id" value="2"/>\n      <metadata key="instance_id" value="0"/>\n'
                '      <metadata key="identify_id" value="1"/>\n    </model_instance>\n  </plate>\n'
                '  <assemble>\n  </assemble>\n</config>\n')
    hexes = ["#" + "".join(format(round(max(0, min(1, c)) * 255), '02X') for c in col) + "FF" for col in colors]
    per = lambda v: [v] * n
    project = {
        "filament_colour": hexes,
        "filament_settings_id": per(f"Bambu PLA Basic @BBL {suffix}"),
        "filament_type": per("PLA"),
        "filament_minimal_purge_on_wipe_tower": per("15"),
        "filament_flow_ratio": per("0.98"),
        "filament_diameter": per("1.75"),
        "wipe": per("1"), "wipe_distance": per("2"),
        "retract_length_toolchange": per("2"), "retract_restart_extra_toolchange": per("0"),
        "standby_temperature_delta": "-5",
        "enable_prime_tower": "1" if n > 1 else "0",
        "prime_tower_width": "20", "prime_tower_brim_width": "3", "prime_volume": "20",
        "flush_into_infill": "0", "flush_into_objects": "0", "flush_into_support": "1",
        "flush_multiplier": "1",
        "flush_volumes_matrix": ["0" if i == j else "280" for i in range(n) for j in range(n)],
        "flush_volumes_vector": ["140"] * (2 * n),
        "wipe_tower_x": ["15"], "wipe_tower_y": ["145" if centre > 100 else "100"],
        "wipe_tower_rotation_angle": "0", "wipe_tower_no_sparse_layers": "0", "wipe_speed": "80%",
        "single_extruder_multi_material": "1", "print_sequence": "by layer",
        "printer_model": model_name,
        "printer_settings_id": f"{model_name} 0.4 nozzle",
        "nozzle_diameter": ["0.4"],
        "print_settings_id": f"0.20mm Standard @BBL {suffix}",
        "layer_change_gcode": "; layer change\nG92 E0\n",
        "from": "project", "name": "project_settings", "version": "02.03.01.00",
    }
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types">\n <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
          'relationships+xml"/>\n <Default Extension="model" ContentType="application/vnd.ms-package.'
          '3dmanufacturing-3dmodel+xml"/>\n</Types>')
    rel = lambda target: ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxml'
                          f'formats.org/package/2006/relationships">\n <Relationship Target="{target}" Id="rel-1" '
                          'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>')
    return [('[Content_Types].xml', ct), ('_rels/.rels', rel('/3D/3dmodel.model')),
            ('3D/_rels/3dmodel.model.rels', rel('/3D/Objects/object_1.model')),
            ('3D/3dmodel.model', main), ('3D/Objects/object_1.model', ''.join(obj)),
            ('Metadata/model_settings.config', settings),
            ('Metadata/project_settings.config', json.dumps(project, indent=4))]
