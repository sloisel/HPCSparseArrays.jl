using Documenter
using MPI

# Initialize MPI for documentation build
if !MPI.Initialized()
    MPI.Init()
end

using LinearAlgebraMPI

makedocs(
    sitename = "LinearAlgebraMPI.jl",
    modules = [LinearAlgebraMPI],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://sloisel.github.io/LinearAlgebraMPI.jl",
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
    repo = "github.com/sloisel/LinearAlgebraMPI.jl.git",
    devbranch = "main",
    push_preview = true,
)
