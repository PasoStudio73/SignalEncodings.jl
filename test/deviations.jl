using Test
using Statistics
using SignalEncodings

@testset "lin_deviation" begin
    @test SignalEncodings.lin_deviation([1, 2, 3]) == 2.0
    @test SignalEncodings.lin_deviation([5, 5, 5, 5]) == 0.0
    @test SignalEncodings.lin_deviation([-2, 0, 2]) == 4.0
    @test SignalEncodings.lin_deviation([1.0, 2.0, 10.0]) ≈
        (abs(1 - 13/3) + abs(2 - 13/3) + abs(10 - 13/3))
end

@testset "sq_deviation" begin
    x = [1, 2, 3]
    @test SignalEncodings.sq_deviation(x) == 2.0

    y = [5, 5, 5, 5]
    @test SignalEncodings.sq_deviation(y) == 0.0

    z = [-2, 0, 2]
    @test SignalEncodings.sq_deviation(z) == 8.0

    # Definition check against population variance
    w = [1.0, 2.0, 10.0, 11.0]
    @test SignalEncodings.sq_deviation(w) ≈ length(w) * var(w; corrected=false)
end