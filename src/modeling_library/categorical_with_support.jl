struct CategoricalWithSupport <: Distribution{Any} end

"""
    categorical(probs::AbstractArray{U, 1}) where {U <: Real}

Given a vector of probabilities `probs` where `sum(probs) = 1`, and a vector of support, sample an element  `support[i]` from the support with probability `probs[i]`.
"""
const categorical_with_support = CategoricalWithSupport()

function logpdf(::CategoricalWithSupport, x::AbstractArray{Any,1}, probs::AbstractArray{U,1},support) where {U <: Real}
    log(probs[x[1]])
end

function logpdf_grad(::CategoricalWithSupport, x::AbstractArray{Any,1}, probs::AbstractArray{U,1},support)  where {U <: Real}
    grad = zeros(length(probs))
    grad[x[1]] = 1.0 / probs[x[1]]
    (nothing, grad)
end

function random(::CategoricalWithSupport, probs::AbstractArray{U,1},support) where {U <: Real}
    idx = rand(Distributions.Categorical(probs))
    [support[idx],idx]
end

(::CategoricalWithSupport)(probs,support) = random(CategoricalWithSupport(), probs,support)

has_output_grad(::CategoricalWithSupport) = false
has_argument_grads(::CategoricalWithSupport) = (true,)

export categorical_with_support