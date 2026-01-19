using Documenter
using MPI

# Initialize MPI for documentation build
if !MPI.Initialized()
    MPI.Init()
end

using HPCSparseArrays

makedocs(
    sitename = "HPCSparseArrays.jl",
    modules = [HPCSparseArrays],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://sloisel.github.io/HPCSparseArrays.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "User Guide" => "guide.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/sloisel/HPCSparseArrays.jl.git",
    devbranch = "main",
    push_preview = true,
)
