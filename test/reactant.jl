@testset "Reactant extension" begin
    if get(ENV, "MFO_TEST_REACTANT", "") == "true" && Base.find_package("Reactant") !== nothing
        include("reactant_parity.jl")
    else
        @test_skip "Reactant parity — run with MFO_TEST_REACTANT=true and Reactant.jl available"
    end
end
