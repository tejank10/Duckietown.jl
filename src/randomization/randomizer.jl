using JSON


mutable struct Randomizer
    randomization_config::Dict
    default_config::Dict
    key_s::Set
end

function Randomizer(randomization_config_fp="default_dr.json", default_config_fp="default.json")
    randomization_config, default_config = nothing, nothing

    try
        open(get_file_path("src/randomization/config", randomization_config_fp, "json"), "r") do f
            randomization_config = json.load(f)
        end
    catch
        #logger.warning("Couldn't find {} in randomization/config subdirectory".format(randomization_config_fp))
        randomization_config = Dict()
    end

    open(get_file_path("src/randomization/config", default_config_fp, "json"), "r") do f
        default_config = JSON.parse(f)
    end

    key_s = Set(append!(collect.(keys.((randomization_config, default_config)))...))

    Randomizer(randomization_config, default_config, key_s)
end

function randomize(randomizer::Randomizer)
    ##Returns a dictionary of randomized parameters, with key: parameter name and value: randomized
    #value
    ##
    randomization_settings = Dict()

    for k in randomizer.key_s
        setting = nothing
        if k in keys(randomizer.randomization_config)
            randomization_definition = randomizer.randomization_config[k]

            if randomization_definition["type"] == "int"
                try
                    low = randomization_definition["low"]
                    high = randomization_definition["high"]
                    size = get(randomization_definition, "size", 1)
                catch
                    throw(KeyError("Please check your randomization definition for: $k"))
                end
                setting = rand(low:high, size)

            elseif randomization_definition["type"] == "uniform"
                try
                    low = randomization_definition["low"]
                    high = randomization_definition["high"]
                    size = get(randomization_definition, "size", 1)
                catch
                    throw(KeyError("Please check your randomization definition for: $k"))
                end

                setting = rand(low:high, size)

            elseif randomization_definition["type"] == "normal"
                try
                    loc = randomization_definition["loc"]
                    scale = randomization_definition["scale"]
                    size = get(randomization_definition, "size", 1)
                catch
                    throw(KeyError("Please check your randomization definition for: $k"))
                end

                setting = rand(Normal(loc, scale), size)

            else
                throw(NotImplementedError("You've specified an unsupported distribution type"))
            end

        elseif k in keys(randomizer.default_config)
            randomization_definition = randomizer.default_config[k]
            setting = randomization_definition["default"]
        end

        randomization_settings[k] = setting
    end
    return randomization_settings
end
