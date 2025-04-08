import warp as wp


@wp.kernel
def subtract_kernel(a: wp.float64, b: wp.float64, out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    result = wp.int(wp.floor(a - b))
    print(a-b)
    print(result)
    out[i] = wp.float64(result)



wp.init()
# Prepare a device array to hold the result:
result = wp.zeros(1, dtype=wp.float64)

# Launch the kernel:
wp.launch(
    kernel=subtract_kernel, 
    dim=1, 
    inputs=[wp.float64(0.2), wp.float64(0.5), result]
)

print(result.numpy())  # should print an array with -0.3
