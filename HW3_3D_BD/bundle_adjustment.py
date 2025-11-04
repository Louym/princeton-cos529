import numpy as np
import pickle
import argparse
loss_tol=2e-3
dist_tol=1e-4
def projection(R, t, f, P):
    """ 
        Project 3D point P using camera parameters R, t, f 
        input: 
            R: (3,3)
            t: (3,)
            f: scalar
            P: (M, 3,)
        output: 
            oiPj: (M, 3)
            projected 2D points (u, v): (M, 2)
    """
    if P.ndim ==1:
        P=P[None,:]
    oipj = P @ R.T + t[None, :]  # Transform point to camera coordinate
    x_proj = f * (oipj[:, 0] / oipj[:, 2])
    y_proj = f * (oipj[:, 1] / oipj[:, 2])
    return oipj, np.stack([x_proj, y_proj], axis=1)

def cal_loss(problem):
    """Rewrite evalation function in my own way, faster since we do not check R"""
    
    poses=problem['poses']
    points=problem['points']
    focal_lengths=problem['focal_lengths']
    observations=problem['observations']
    
    M=points.shape[0]
    N=focal_lengths.shape[0]
    errors=np.zeros((M*N, 2))
    # total_loss=0
    for index, (cam_i, point_j, x, y) in enumerate(observations):
        fi=focal_lengths[cam_i]
        Ri=poses[cam_i][:3,:3]
        ti=poses[cam_i][0:3,-1]
        Pij1=projection(Ri, ti, fi, points[point_j])[-1]
        (u, v)=Pij1[0]
        errors[cam_i*M + point_j, 0]=u-x
        errors[cam_i*M + point_j, 1]=v-y
        # total_loss+=np.sum(errors[index]**2)
    return np.mean(np.sum(errors ** 2, axis=1)**0.5), errors.reshape(-1)

def batch_cross_matrix(A):
    """Compute cross-product matrices for a batch of 3D vectors ."""
    ax, ay, az = A[:, 0], A[:, 1], A[:, 2]
    zeros = np.zeros_like(ax)
    return np.stack([
        np.stack([zeros, -az, ay], axis=-1),
        np.stack([az, zeros, -ax], axis=-1),
        np.stack([-ay, ax, zeros], axis=-1)
    ], axis=1)

def construct_R(posai):
    """Axis-angle (N,3) → Rotation matrix (N,3,3)."""
    theta = np.linalg.norm(posai, axis=1, keepdims=True)
    phai = posai / (theta + 1e-10)
    cos_theta = np.cos(theta)[:, None]
    sin_theta = np.sin(theta)[:, None]
    cross_matrix = batch_cross_matrix(phai)

    R = (
        cos_theta * np.eye(3)
        + (1 - cos_theta) * (phai[:, :, None] @ phai[:, None, :])
        + sin_theta * cross_matrix
    )
    return R

def get_posai(R):
    """Rotation matrix (N,3,3) → Axis-angle (N,3)."""
    theta = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))  # (N,)
    phai = np.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], axis=1)  # (N,3)
    denom = 2 * np.sin(theta)[:, None]
    phai = phai / (denom + 1e-12)
    posai = theta[:, None] * phai
    return posai

def update_parameters(problem, update):
    """ 
        Update the parameters of the problem using the update vector
        input:
            problem
            update: shape (6*N+3*M+(0 if is_calibrated else N), 1)
        output:
            problem: updated problem
    """
    if problem['is_calibrated']:
        camera_parameter_num=6
    else:
        camera_parameter_num=7
    N=problem['focal_lengths'].shape[0]
    M=problem['points'].shape[0]
    for i in range(N):
        delta_camera=update[camera_parameter_num*i:camera_parameter_num*(i+1)]
        delta_posai=delta_camera[0:3]
        delta_t=delta_camera[3:6]
        Ri=problem['poses'][i][:3,:3]
        ti=problem['poses'][i][0:3,-1]
        posai=get_posai(Ri[None, :, :])[0]
        new_posai=posai+delta_posai
        new_Ri=construct_R(new_posai[None, :])[0]
        new_ti=ti+delta_t
        problem['poses'][i][:3,:3]=new_Ri
        problem['poses'][i][0:3,-1]=new_ti
        if not problem['is_calibrated']:
            delta_f=delta_camera[6]
            problem['focal_lengths'][i]+=delta_f
    for j in range(M):
        delta_point=update[camera_parameter_num*N+3*j:camera_parameter_num*N+3*j+3]
        problem['points'][j]+=delta_point
    return problem


def get_JTJ_JTe(solution, return_list=False):
    """ 
        Directly get JTJ and JTb to save time and memory
        Since some observations may be missing, we need to loop over all observations rather than cameras or points
        input:
            solution
        output:
            J.T @ J, shape (6*N+3*M+(0 if is_calibrated else N), 6*N+3*M+(0 if is_calibrated else N))
            J.T @ errors, shape (6*N+3*M+(0 if is_calibrated else N), 1)
    """
    M=solution['points'].shape[0]
    N=solution['focal_lengths'].shape[0]
    observations=solution['observations']
    if solution['is_calibrated']:
        camera_parameter_num=6
    else:
        camera_parameter_num=7
    # Initialize
    # e
    errors=np.zeros(2*len(observations))
    # J_patch
    J_camera_patch= np.zeros((2, camera_parameter_num))
    J_points_patch= np.zeros((2, 3))
    # JTe 
    JTe= np.zeros((camera_parameter_num*N+3*M,1))
    # JTT=[[A, C],[C.T, B]] and A, B is block diagonal matrix
    if return_list:
        A=[np.zeros((camera_parameter_num, camera_parameter_num)) for _ in range(N)]
        B=[np.zeros((3, 3)) for _ in range(M)]
    else:
        A=np.zeros((camera_parameter_num*N, camera_parameter_num*N))
        B=np.zeros((3*M, 3*M))
    C=np.zeros((camera_parameter_num*N, 3*M))
    for index, (cam_i, point_j, x, y) in enumerate(observations):
        fi=solution['focal_lengths'][cam_i]
        Ri=solution['poses'][cam_i][:3,:3]
        ti=solution['poses'][cam_i][0:3,-1]
        # Compute oipj and errors
        oipj, uv=projection(Ri, ti, fi, solution['points'][point_j])
        u, v=uv[0]
        errors[index*2]=u-x
        errors[index*2+1]=v-y
        error_patch=np.array([u-x, v-y])
        # Compute J_camera_patch
        de_dt=np.array([
                [fi/oipj[0,2], 0, -fi*oipj[0,0]/(oipj[0,2]**2)],
                [0, fi/oipj[0,2], -fi*oipj[0,1]/(oipj[0,2]**2)]
            ])  # (2, 3)
        # de_dposai=de_doipj @ doipj_dposai, de_doipj=de_dt
        doipj_dposai=-Ri@batch_cross_matrix(solution['points'][point_j][None, :])  # (1, 3, 3)
        de_dposai=(de_dt @ doipj_dposai[0])  # (2, 3)
        J_camera_patch[:, 0:3]=de_dposai
        J_camera_patch[:, 3:6]=de_dt
        if not solution['is_calibrated']:
            de_df=np.array([
                    [oipj[0,0]/oipj[0,2]],
                    [oipj[0,1]/oipj[0,2]]
                ])  # (2, 1)
            J_camera_patch[:, 6]=de_df[:,0]
        # Compute J_points_patch
        doipj_dpj=Ri # (3, 3)
        J_points_patch=de_dt @ doipj_dpj  # (2, 3)        

        offset_camera=cam_i*camera_parameter_num
        offset_points=camera_parameter_num*N+point_j*3
        
        # Update JTe
        JTe[offset_camera:camera_parameter_num+offset_camera,:]+=J_camera_patch.T @ error_patch.reshape(-1, 1)
        JTe[offset_points:offset_points+3,:]+=J_points_patch.T @ error_patch.reshape(-1, 1)

        # Update JTJ
        if not return_list:
            # Update A
            A[cam_i*camera_parameter_num:cam_i*camera_parameter_num+camera_parameter_num, cam_i*camera_parameter_num:cam_i*camera_parameter_num+camera_parameter_num]+= J_camera_patch.T @ J_camera_patch
            # Update B
            B[point_j*3:point_j*3+3, point_j*3:point_j*3+3]+= J_points_patch.T @ J_points_patch
        else:
            # Update A
            A[cam_i]+= J_camera_patch.T @ J_camera_patch
            # Update B
            B[point_j]+= J_points_patch.T @ J_points_patch
        # Update C
        row=camera_parameter_num*cam_i
        col=point_j*3
        C[row:row+camera_parameter_num, col:col+3]+= J_camera_patch.T @ J_points_patch
    return A, B, C, JTe.reshape(-1)

def block_diag(blocks):
    """ 
        Create a block diagonal matrix from a list of blocks
        input:
            blocks: list of 2D numpy arrays
        output:
            block_diag: 2D numpy array
    """
    return np.block([[block if i == j else np.zeros_like(block) for j, block in enumerate(blocks)] for i, _ in enumerate(blocks)])

def inverse_B(B):
    """ 
        Inverse of block diagonal matrix B
        input:
            B: block diagonal matrix, shape (3*M, 3*M)
        output:
            B_inv: inverse of B, shape (3*M, 3*M)
    """
    if isinstance(B, list):
        return [np.linalg.inv(block) for block in B]
    M=B.shape[0]//3
    B_inv=np.zeros((3*M, 3*M))
    for j in range(M):
        B_inv[3*j:3*j+3, 3*j:3*j+3]=np.linalg.inv(B[3*j:3*j+3, 3*j:3*j+3])
    return B_inv
def block_diag_mat(C, list_B):
    """ 
        Multiply C with block diagonal matrix B
        input:
            C: shape (camera_parameter_num*N, 3*M)
            list_B: list of blocks of B, each block shape (3, 3)
        output:
            C_B: shape (camera_parameter_num*N, 3*M)
    """
    camera_parameter_num=C.shape[0]//len(list_B)
    M=len(list_B)
    C_B=np.zeros_like(C)
    for j in range(M):
        block_B=list_B[j]
        C_B[:, 3*j:3*j+3]=C[:, 3*j:3*j+3] @ block_B
    return C_B
def block_diag_Levenberg_Marquardt(A, lambda_, epsilon_):
    """ 
        Add Levenberg-Marquardt damping to block diagonal matrix A
        input:
            list_A: list of blocks of A, or np.array
            lambda_: damping factor
        output:
            A_damped: block diagonal matrix with damping, shape (camera_parameter_num*N, camera_parameter_num*N)
    """
    if isinstance(A, list):
        return [ a + lambda_ * np.diag(np.diag(a)) + epsilon_ * np.eye(a.shape[0]) for a in A]
    else:
        A[np.diag_indices_from(A)] *= (1 + lambda_)
        A[np.diag_indices_from(A)] += epsilon_
        return A

def schur_complement_solve(A, B, C, b, M, N, calibrated):
    camera_parameter_num = 6 if calibrated else 7
    update = np.zeros((camera_parameter_num*N+3*M))
    # Compute Schur complement
    B_inv=inverse_B(B)
    CB_inv=C@B_inv
    A1=A-CB_inv@C.T
    b1=b[0:camera_parameter_num*N]-CB_inv@b[camera_parameter_num*N:camera_parameter_num*N+3*M]
    # Solve for camera update
    update[0:camera_parameter_num*N] = np.linalg.solve(A1, b1)
    # Compute point update
    b2=b[camera_parameter_num*N:camera_parameter_num*N+3*M]-C.T @ update[0:camera_parameter_num*N]
    # Solve for update
    update[camera_parameter_num*N:camera_parameter_num*N+3*M] = B_inv@b2
    return update
    

def solve_ba_problem_basic(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''

    solution = problem
    # YOUR CODE STARTS
    M=solution['points'].shape[0]
    N=solution['focal_lengths'].shape[0]
    print(f"Solving BA problem with N={N} cameras and M={M} points.")
    # e, e1=cal_loss(solution)
    # print(e)
    # Initialize
    parameter_lambda=1e-6
    parameter_epsilon=1e-6
    init_loss, _=cal_loss(solution)      
    while(True):
        ## Optimize
        # Compute Jacobian matrix
        # 
        A, B, C, b = get_JTJ_JTe(solution)
        A=np.block([
            [A, C],
            [C.T, B]
        ])

        # Solve
        A = A + parameter_lambda * np.diag(np.diag(A)) + parameter_epsilon * np.eye(A.shape[0])
        update = np.linalg.solve(A, -b)
        
        # Update parameters
        solution=update_parameters(solution, update)
        
        # Judge update
        current_loss, current_errors=cal_loss(solution)
        print(current_loss)
        loss_improvement=init_loss-current_loss
        if loss_improvement<0:
            # Failed update, roll back
            solution=update_parameters(solution, -update)
            parameter_lambda=parameter_lambda*2
        else:
            # Successful update
            parameter_lambda=parameter_lambda/2
            init_loss=current_loss
            
        # Judge convergence
        if abs(loss_improvement)<loss_tol and loss_improvement>=0:
            break
        dist=np.linalg.norm(update)
        if dist<dist_tol:
            break
        
    

    return solution

def solve_ba_problem(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''

    solution = problem
    # YOUR CODE STARTS
    M=solution['points'].shape[0]
    N=solution['focal_lengths'].shape[0]
    # print(f"Solving BA problem with N={N} cameras and M={M} points.")
    parameter_lambda=1e-5
    parameter_epsilon=1e-4
    # Initialize
    init_loss, _=cal_loss(solution)      
    while(True):
        ## Optimize
        # Compute Jacobian matrix
        # 
        A, B, C, b = get_JTJ_JTe(solution)
        # This is very memory-consuming since B will be very large
        # A = A + parameter_lambda * np.diag(np.diag(A)) + parameter_epsilon * np.eye(A.shape[0])
        # B = B + parameter_lambda * np.diag(np.diag(B)) + parameter_epsilon * np.eye(B.shape[0])
        # So we use in-place operation in block_diag_Levenberg_Marquardt
        A=block_diag_Levenberg_Marquardt(A, parameter_lambda, parameter_epsilon)
        B=block_diag_Levenberg_Marquardt(B, parameter_lambda, parameter_epsilon)
        update= -schur_complement_solve(A, B, C, b, M, N, solution['is_calibrated'])
        
        # Update parameters
        solution=update_parameters(solution, update)
        
        # Judge update
        current_loss, current_errors=cal_loss(solution)
        # print(current_loss)
        loss_improvement=init_loss-current_loss
        if loss_improvement<0:
            # Failed update, roll back
            solution=update_parameters(solution, -update)
            parameter_lambda=parameter_lambda*2
        else:
            # Successful update
            parameter_lambda=parameter_lambda/2
            init_loss=current_loss
            
        # Judge convergence
        if abs(loss_improvement)<loss_tol and loss_improvement>=0:
            break
        dist=np.linalg.norm(update)
        if dist<dist_tol:
            break
        
    

    return solution
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()
    problem = pickle.load(open(args.problem, 'rb'))
    solution = solve_ba_problem(problem)
    
    # For time and memory profiling
    # import time
    # import tracemalloc

    # tracemalloc.start()  
    # start_time = time.perf_counter()
    # problem = pickle.load(open(args.problem, 'rb'))
    # solution = solve_ba_problem(problem)
    # end_time = time.perf_counter()
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Peak memory usage: {peak / 1024**2:.2f} MB")

    # # For time profiling
    # print(f"Time taken: {end_time - start_time:.2f} seconds")

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
