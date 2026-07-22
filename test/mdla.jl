@testset "MDLA distributed" begin
    if get(ENV, "MFO_TEST_MDLA", "") == "true" &&
       Base.find_package("MultiDeviceLinearAlgebra") !== nothing &&
       Base.find_package("CUDA") !== nothing
        include("mdla_gpu.jl")
    else
        @test_skip "MDLA distributed — run with MFO_TEST_MDLA=true, CUDA.jl and MultiDeviceLinearAlgebra.jl available"
    end
end
