"Reshapes a `(r,c,n)` array into `(r*c, n)`."
pattern_to_vector(P::Array{T,3}) where T = 2*(reshape(copy(P), prod(size(P)[1:2]), size(P,3))) .-1
pattern_to_vector(P::Matrix{T}) where T = 2*(reshape(copy(P), prod(size(P)[1:2]))) .-1

"Reshapes a vector of length `r*c` into `(r,c)` matrix."
function vector_to_pattern(v::Vector; r=Int(sqrt(length(v))))
    c = length(v)÷r
    return copy(reshape( (v.+1).÷2, r, c))
end