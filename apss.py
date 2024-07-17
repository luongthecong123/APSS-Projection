import numpy as np
# from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import trimesh

class APSS:
    MLS_TOO_FAR = -1
    MLS_OK = 0
    MLS_TOO_MANY_ITERS = -2
    MLS_DERIVATIVE_ACCURATE = 1
    ASS_SPHERE = 0
    ASS_PLANE = 1
    ASS_UNDETERMINED = 2

    def __init__(self, mesh_points, mesh_normals, average_point_spacing, projection_accuracy, max_nof_projection_iterations):
        '''
        1. self.mSphericalParameter:
        Increasing this value makes the MLS fit more sensitive to curvature, favoring spherical shapes. Decreasing it favors planar fits.
        
        2. self.uQuad:
        Increasing this value introduces more curvature, making the fitted surface more sphere-like. A zero value results in a planar fit.
        '''

        self.mesh_points = np.array(mesh_points)
        self.mesh_normals = np.array(mesh_normals)
        self.average_point_spacing = average_point_spacing
        self.projection_accuracy = projection_accuracy
        self.max_nof_projection_iterations = max_nof_projection_iterations

        # self.kdtree = KDTree(self.mesh_points)
        self.kdtree = cKDTree(self.mesh_points)
        self.mSphericalParameter = 0.0
        self.mCachedQueryPointIsOK = False
        self.mCachedQueryPoint = None
        self.uLinear = np.zeros(3)
        self.uConstant = 0.0
        self.uQuad = 0.0
        self.mStatus = self.ASS_UNDETERMINED
        self.mCenter = np.zeros(3)
        self.mRadius = 0.0

    def project(self, x, pNormal=None, errorMask=None):
        iterationCount = 0
        lx = x
        position = lx
        normal = np.zeros(3)
        previousPosition = np.zeros(3)
        delta2 = 0
        epsilon2 = self.average_point_spacing * self.projection_accuracy
        epsilon2 *= epsilon2
        while True:
            if not self.fit(position):
                if errorMask is not None:
                    errorMask[0] = self.MLS_TOO_FAR
                return x

            previousPosition = position.copy()
            if self.mStatus == self.ASS_SPHERE:
                normal = lx - self.mCenter
                normal /= np.linalg.norm(normal)
                position = self.mCenter + normal * self.mRadius

                normal = self.uLinear + position * (2 * self.uQuad)
                normal /= np.linalg.norm(normal)
            elif self.mStatus == self.ASS_PLANE:
                normal = self.uLinear
                position = lx - self.uLinear * (np.dot(lx, self.uLinear) + self.uConstant)
            else:
                grad = self.uLinear + lx * (2 * self.uQuad)
                dir = grad / np.linalg.norm(grad)
                ad = self.uConstant + np.dot(self.uLinear, lx) + self.uQuad * np.dot(lx, lx)
                delta = -ad * min(1.0, 1.0 / np.linalg.norm(grad))
                position = lx + dir * delta

                for _ in range(2):
                    grad = self.uLinear + position * (2 * self.uQuad)
                    delta = -(self.uConstant + np.dot(self.uLinear, position) + self.uQuad * np.dot(position, position)) * min(1.0, 1.0 / np.linalg.norm(grad))
                    position += dir * delta

                normal = self.uLinear + position * (2 * self.uQuad)
                normal /= np.linalg.norm(normal)

            delta2 = np.dot(previousPosition - position, previousPosition - position)
            if delta2 <= epsilon2 or iterationCount >= self.max_nof_projection_iterations:
                break
            iterationCount += 1

        if pNormal is not None:
            if self.mGradientHint == self.MLS_DERIVATIVE_ACCURATE:
                grad = np.zeros(3)
                self.mlsGradient(position, grad)
                grad /= np.linalg.norm(grad)
                pNormal[:] = grad
            else:
                pNormal[:] = normal

        if iterationCount >= self.max_nof_projection_iterations and errorMask is not None:
            errorMask[0] = self.MLS_TOO_MANY_ITERS

        return position

    def fit(self, x):
        self.computeNeighborhood(x, True)
        nofSamples = len(self.neighborhood)

        if nofSamples == 0:
            self.mCachedQueryPointIsOK = False
            return False
        elif nofSamples == 1:
            id = self.neighborhood[0]
            p = self.mesh_points[id]
            n = self.mesh_normals[id]

            self.uLinear = n
            self.uConstant = -np.dot(p, self.uLinear)
            self.uQuad = 0
            self.mStatus = self.ASS_PLANE
            return True

        sumP = np.zeros(3)
        sumN = np.zeros(3)
        sumDotPN = 0.0
        sumDotPP = 0.0
        sumW = 0.0

        for i in range(nofSamples):
            id = self.neighborhood[i]
            p = self.mesh_points[id]
            n = self.mesh_normals[id]
            w = self.cached_weights[i]

            sumP += p * w
            sumN += n * w
            sumDotPN += w * np.dot(n, p)
            sumDotPP += w * np.dot(p, p)
            sumW += w

        invSumW = 1.0 / sumW
        aux4 = self.mSphericalParameter * 0.5 * (sumDotPN - invSumW * np.dot(sumP, sumN)) / (sumDotPP - invSumW * np.dot(sumP, sumP))
        self.uLinear = (sumN - sumP * (2.0 * aux4)) * invSumW
        self.uConstant = -invSumW * (np.dot(self.uLinear, sumP) + sumDotPP * aux4)
        self.uQuad = aux4

        if abs(self.uQuad) > 1e-7:
            self.mStatus = self.ASS_SPHERE
            b = 1.0 / self.uQuad
            self.mCenter = -self.uLinear * 0.5 * b
            self.mRadius = np.sqrt(np.dot(self.mCenter, self.mCenter) - b * self.uConstant)
        elif self.uQuad == 0.0:
            self.mStatus = self.ASS_PLANE
        else:
            self.mStatus = self.ASS_UNDETERMINED

        self.mCachedQueryPoint = x
        self.mCachedQueryPointIsOK = True

        return True

    def computeNeighborhood(self, x, recomputeWeights):
        distances, indices = self.kdtree.query([x], k=10)
        self.neighborhood = indices[0]
        self.cached_weights = np.exp(-distances[0]**2 / (2 * self.average_point_spacing**2)) if recomputeWeights else self.cached_weights


def load_mesh(file_path):
    mesh = trimesh.load_mesh(file_path, process=False)
    return mesh.vertices, mesh.faces, mesh.vertex_normals

def save_mesh(file_path, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(file_path)

def main():
    processed_mesh_path = 'mesh/DFAUST_f_1024_subdiv.ply'
    raw_mesh_path = 'mesh/DFAUST_f_1024.ply'
    output_mesh_path = 'mesh/Exp_sphere0.0_spacing0.01_proj0.0001_iter10_rm_useless.ply'

    # Load processed and raw meshes
    processed_vertices, processed_faces, processed_normals = load_mesh(processed_mesh_path)
    raw_vertices, _, raw_normals = load_mesh(raw_mesh_path)

    # Initialize APSS with the raw mesh
    average_point_spacing = 0.01
    projection_accuracy = 0.0001
    max_nof_projection_iterations = 10
    apss = APSS(raw_vertices, raw_normals, average_point_spacing, projection_accuracy, max_nof_projection_iterations)

    # Project processed mesh vertices onto raw mesh
    projected_vertices = []
    for vertex in processed_vertices:
        projected_vertex = apss.project(vertex)
        projected_vertices.append(projected_vertex)

    projected_vertices = np.array(projected_vertices)

    # Save the projected mesh
    save_mesh(output_mesh_path, projected_vertices, processed_faces)

if __name__ == '__main__':
    import time

    for i in range(10):
        s = time.time()
        main()
        e = time.time()
        print("Iter: ",i , "Total execution time: ", e-s)
