using Documenter
using SignalEncodings

DocMeta.setdocmeta!(
    SignalEncodings,
    :DocTestSetup,
    :(using SignalEncodings);
    recursive = true
)

makedocs(;
    modules=[SignalEncodings],
    authors="Riccardo Pasini",
    repo=Documenter.Remotes.GitHub("PasoStudio73", "SignalEncodings.jl"),
    sitename="SignalEncodings.jl",
    format=Documenter.HTML(;
        size_threshold=4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PasoStudio73.github.io/SignalEncodings.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Binning" => "bin.md",
        "Algorithms" => "algos.md"
    ],
    warnonly=true,
)

deploydocs(;
    repo = "github.com/PasoStudio73/SignalEncodings.jl",
    devbranch = "main",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
