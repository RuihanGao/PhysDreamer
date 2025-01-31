import warp as wp

wp.init()

# Initialize Warp arrays
x = wp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=wp.vec3)  # Shape: (1, 3)
v = wp.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=wp.vec3)  # Shape: (1, 3)
x2 = wp.zeros_like(x)  # Output array with the same shape as x

dt = float(1)  # Scalar


# wp.kernel()
# def sum_vec3(x: wp.array(dtype=wp.vec3), y: wp.array(dtype=wp.vec3), z: wp.array(dtype=float)):
#     tid = wp.tid()
#     wp.atomic_add(z, 0, x[tid][0]*y[tid][0] + x[tid][1]*y[tid][1] + x[tid][2]*y[tid][2])

# Define a Warp kernel
@wp.kernel
def update_position(x: wp.array(dtype=wp.vec3),
                    v: wp.array(dtype=wp.vec3),
                    dt: float):

    i = wp.tid()  # Thread index

    # Ensure we access elements correctly from a 2D array
    # x2[i][0] = x[i][0] + v[i][0] * dt
    # x2[i][1] = x[i][1] + v[i][1] * dt
    # x2[i][2] = x[i][2] + v[i][2] * dt

    # wp.atomic_add(x2[i], 0,  x[i][0] + v[i][0] )
    # wp.atomic_add(x2[i], 1,  x[i][1] + v[i][1] )
    # wp.atomic_add(x2[i], 2,  x[i][2] + v[i][2] )

    wp.atomic_add(x, i,  v[i]*dt)

# Launch the kernel
wp.launch(kernel=update_position,
          dim=x.shape[0],  # Process all elements in the arrays
          inputs=[x, v, dt,])

# Convert to NumPy for verification
print("x:", x.numpy())