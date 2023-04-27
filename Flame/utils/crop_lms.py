from pytorch3d.io import load_objs_as_meshes, load_obj
import torch
import trimesh

def get_face_inds():
    v_t, f_t, a_t = load_obj('Flame/data/head_template_mesh.obj')

    #v_c, f_c, a_c = load_obj('Flame/data/head_template_mesh_cropped_nostrils.obj')
    v_c, f_c, a_c = load_obj('Flame/data/head_template_mesh_cropped_nostrils_test.obj')

    template = trimesh.load('Flame/data/head_template_mesh.obj')
    cropped = trimesh.load('Flame/data/head_template_mesh_cropped_nostrils_test.obj')
    #cropped.show()
    #template.show()

    # mat, trans = trimesh.registration.mesh_other(cropped, template)
    # cropped = cropped.apply_transform(mat)
    # cropped.export('test.obj')
    

    # template = load_objs_as_meshes(['Flame/data/head_template_mesh.obj'], load_textures=False)
    # template = template.verts_list()[0]
    # cropped_face = load_objs_as_meshes(['Flame/data/head_template_mesh_cropped.obj'], load_textures=False)
    # cropped_face = cropped_face.verts_list()[0]
    # cropped_nostrils = load_objs_as_meshes(['Flame/data/head_template_mesh_cropped_nostrils.obj'], load_textures=False)
    # cropped_nostrils = cropped_nostrils.verts_list()[0]
    # cropped_eyes = load_objs_as_meshes(['Flame/data/head_template_mesh_cropped_nostrils_eyes.obj'], load_textures=False)
    # cropped_eyes = cropped_eyes.verts_list()[0]

    values, indices = torch.topk(((v_t.t() == v_c.unsqueeze(-1)).all(dim=1)).int(), 1, 1)
    indices = indices[values!=0]

    return indices
# indices = tensor([0, 2])

if __name__ == '__main__':
    get_face_inds()