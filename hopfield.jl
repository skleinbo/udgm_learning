import Distributions: Bernoulli
import LinearAlgebra: mul!, muladd
import Flux.Losses: logitbinarycrossentropy
using Optimisers
using Zygote

include("util.jl")

abstract type AbstractHopfieldNetwork end

activations(H::AbstractHopfieldNetwork) = muladd(H.W, H.x, H.θ)
activation(H::AbstractHopfieldNetwork, i) = @views dot(H.W[i, :], H.x) + H.θ[i]
    
struct BinaryHopfieldNetwork <: AbstractHopfieldNetwork
    W::Matrix{Float64}
    θ::Vector{Float64}
    x::Vector{Int}
end
BinaryHopfieldNetwork(N::Int) = BinaryHopfieldNetwork(2*(rand(N,N).-1), zeros(Float64, N), rand([-1,1], N))
# (H::HopfieldNetwork)(x) = Int.(sign.(H.W*x .- H.θ))

struct ContinuousHopfieldNetwork <: AbstractHopfieldNetwork
    W::Matrix{Float64}
    θ::Vector{Float64}
    x::Vector{Float64}
end
ContinuousHopfieldNetwork(N::Int) = ContinuousHopfieldNetwork(2*rand(N,N).-1, zeros(Float64, N), 2*rand(N).-1)

function (H::ContinuousHopfieldNetwork)(x; y=nothing)
    if isnothing(y)
        y = similar(x)
    end
    y .= tanh.(activations(H))
    return y
end 
    
function (H::BinaryHopfieldNetwork)(x; y=nothing)
    if isnothing(y)
        y = similar(x)
    end
    for i in eachindex(y)
        y[i] = Int(sign(activation(H, i)))
    end
    return y
end

chn(W::Matrix, θ::Vector,  x) = begin
    a = muladd(W, x, θ)
    return (a, tanh.(a))
end

function recall(W, θ, seed; maxiter=100)
    iter = 0
    x = copy(seed)
    a = similar(x)
    while true
        a, x = chn(W, θ, seed)
        if isapprox(x,seed) || iter > maxiter
            break
        end
        iter += 1
    end
    return (a,x)
end
        
step!(H::AbstractHopfieldNetwork) = H(H.x, y=H.x) 

energy(H::AbstractHopfieldNetwork, x) = -1/2*dot(x, H.W, x)
energy(H::AbstractHopfieldNetwork) = energy(H, H.x)

overlap(x,y) = dot(x, y)/length(x)
overlap(X::Array{T, 2}, Y::Array{T, 2}) where T = overlap.(eachcol(X), eachcol(Y))
overlap(H::AbstractHopfieldNetwork, x) = overlap(H.x, x)

function train_hebbe!(H::AbstractHopfieldNetwork, patterns::Matrix)
    # H.W .= 0
    # for pattern in eachcol(patterns)
    mul!(H.W, patterns, patterns')
    # end
    H.W ./= size(patterns,2)
    for i in axes(H.W, 2)
        H.W[i,i] = 0.0
    end
    nothing
end

is_converged(H::BinaryHopfieldNetwork) = H(H.x) == H.x
is_converged(H::ContinuousHopfieldNetwork) = isapprox(H(H.x), H.x)


function train_map!(H::ContinuousHopfieldNetwork, patterns::Matrix; J=0.0, pretrain=false, bias=false, p_noise=0.01, epochs=1)
    patterns_01 = (patterns.+1).÷2
    # Pre-train
    pretrain && train_hebbe!(H, patterns)
    # Fine-tune
    model = (W=H.W, θ=H.θ)
    tree = Optimisers.setup(Adam(), model)
    if !bias
        Optimisers.freeze!(tree.θ)
    end
    for epoch in 1:epochs
        # randomly flip bits
        pattern_masked = copy(patterns)
        pattern_masked .+= rand(Bernoulli(p_noise), size(pattern_masked))
        pattern_masked .%= 2

        for i in size(model.W, 1)
            model.W[i,i] = 0.0
        end
        
        loss, gs = withgradient(model) do m
            pred_a, pred_x = recall(m.W, m.θ, pattern_masked)
            logitbinarycrossentropy(pred_a, patterns_01)
        end
        gs = (;gs[1]..., W=(gs[1].W+gs[1].W')./2)
        Optimisers.update!(tree, model, gs)
        (epoch==epochs || epoch%10==0) && @show epoch, loss
    end
    return H
end

function recall!(H::AbstractHopfieldNetwork, s::Vector; maxiter=30, verbose=false)
    iter = 0
    H.x .= s
    verbose && @show iter, energy(H, H.x)
    while !is_converged(H) && iter < maxiter
        step!(H)
        iter += 1
        verbose && @show iter, energy(H, H.x)
    end
    return (H.x, iter, iter<maxiter)
end
function recall!(H::AbstractHopfieldNetwork, seeds; kwargs...)
    mapslices(seeds, dims=1) do seed
        recall!(H, seed; kwargs...)[1]
    end
end

## Recall per pattern
function fraction_sample(pattern, recalls)
    reshape(mapslices(pattern .== recalls, dims=(1,2)) do x
        count(x) / prod(size(pattern)[1:2])
    end, :)
end

function onezero(pattern, recalls)
    reshape(map(axes(pattern, 3)) do i
        pattern[i] == recalls[i]
    end, :)
end

"""
    measure_recall_random(n, [nsamples=1:n]; [loss=overlap])

Generates random `n^2` 0-1 samples, trains a Hopfield network on them, and
returns a loss for each.
"""
function measure_recall_random(n, nsamples=1; H=nothing, p_noise=0.01, loss=overlap, kwargs...)
    if isnothing(H)
        H = BinaryHopfieldNetwork(n*n)
    end
    
    T = eltype(H.W)
    pattern = rand(T[-1,1], n^2, nsamples)

    pattern_masked = copy(pattern)
    # pattern_masked[1:1*size(pattern_masked,1)÷2,:] .= -1
    pattern_masked .+= rand(Bernoulli(p_noise), size(pattern_masked))
    pattern_masked .%= 2

    train_hebbe!(H, pattern)

    recalls = similar(pattern)
    converged = Bool[]
    iters = Int[]
    ProgressMeter.@showprogress for i in axes(pattern_masked, 2)
        s = pattern_masked[:,i]
        (x, iter, conv) = recall!(H, s; kwargs...)
        push!(converged, conv)
        push!(iters, iter)
        recalls[:,i] .= x
    end
    # @show typeof(pattern) typeof(recalls)
    loss(pattern, recalls), converged, iters
end


