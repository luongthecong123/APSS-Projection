import numpy as np
from scipy.spatial import cKDTree
import trimesh

class APSS:
    def __init__(self, mesh_points, mesh_normals, average_point_spacing, projection_accuracy, max_nof_projection_iterations):

        self.mesh_points = np.array(mesh_points, dtype=np.float64)
        self.mesh_normals = np.array(mesh_normals, dtype=np.float64)
        # self.mesh_points = mesh_points
        # self.mesh_normals = mesh_normals
        self.average_point_spacing = average_point_spacing
        self.projection_accuracy = projection_accuracy
        self.max_nof_projection_iterations = max_nof_projection_iterations
        self.kdtree = cKDTree(self.mesh_points)

    def project_batch(self, vertices):
        epsilon2 = (self.average_point_spacing * self.projection_accuracy) ** 2
        
        distances, indices = self.kdtree.query(vertices, k=10)
        weights = np.exp(-distances**2 / (2 * self.average_point_spacing**2))
        
        projected_vertices = []
        
        for i, vertex in enumerate(vertices):
            position = vertex
            for _ in range(self.max_nof_projection_iterations):
                neighbor_points = self.mesh_points[indices[i]]
                neighbor_normals = self.mesh_normals[indices[i]]
                w = weights[i]
                
                sumW = np.sum(w)
                if sumW == 0:
                    projected_vertices.append(vertex)
                    break
                
                sumP = np.sum(neighbor_points * w[:, np.newaxis], axis=0)
                sumN = np.sum(neighbor_normals * w[:, np.newaxis], axis=0)
                
                invSumW = 1.0 / sumW
                uLinear = sumN * invSumW
                uConstant = -np.dot(uLinear, sumP * invSumW)
                
                new_position = position - uLinear * (np.dot(position, uLinear) + uConstant)
                
                if np.sum((new_position - position)**2) <= epsilon2:
                    projected_vertices.append(new_position)
                    break
                
                position = new_position
            else:
                projected_vertices.append(position)
        
        return np.array(projected_vertices)

def load_mesh(file_path):
    mesh = trimesh.load_mesh(file_path, process=False)
    return mesh.vertices, mesh.faces, mesh.vertex_normals

def save_mesh(file_path, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(file_path)

def main():
    processed_mesh_path = 'mesh/DFAUST_f_1024_subdiv.ply'
    raw_mesh_path = 'mesh/DFAUST_f_1024.ply'
    output_mesh_path = 'mesh/Exp_sphere0.0_spacing0.01_proj0.0001_iter10_Numba.ply'

    processed_vertices, processed_faces, _ = load_mesh(processed_mesh_path)
    raw_vertices, _, raw_normals = load_mesh(raw_mesh_path)

    average_point_spacing = 0.01
    projection_accuracy = 0.0001
    max_nof_projection_iterations = 10
    apss = APSS(raw_vertices, raw_normals, average_point_spacing, projection_accuracy, max_nof_projection_iterations)

    projected_vertices = apss.project_batch(processed_vertices)

    save_mesh(output_mesh_path, projected_vertices, processed_faces)

if __name__ == '__main__':
    import time

    total_s = time.time()
    for i in range(10):
        s = time.time()
        main()
        e = time.time()
        print("Iter: ", i, "Execution time: ", e-s)
    total_e = time.time()

    print("Total execution time: ", total_e - total_s)