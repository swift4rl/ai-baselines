#!/usr/bin/env ruby

require 'xcodeproj'

task :dependencies do

  system "swift package update"
  system "swift package generate-xcodeproj"

  project = Xcodeproj::Project.open('ai-baselines.xcodeproj')

  project.targets.each do |target|
    project.build_configurations.each { |config|
      config.build_settings["MACOSX_DEPLOYMENT_TARGET"] = 11.0
      config.build_settings["ONLY_ACTIVE_ARCH"] = 'YES'
      config.build_settings["SUPPORTED_PLATFORMS"] = 'macosx'
      config.build_settings["LD_RUNPATH_SEARCH_PATHS"] = "$(inherited) @executable_path/Frameworks"
      config.build_settings["SWIFT_VERSION"] = '5.3'
      config.build_settings["ENABLE_TESTING_SEARCH_PATHS"] = 'YES'
      config.build_settings.delete("FRAMEWORK_SEARCH_PATHS")
    }

    target.build_configurations.each do |config|
      config.build_settings["MACOSX_DEPLOYMENT_TARGET"] = 11.0
      config.build_settings["SUPPORTED_PLATFORMS"] = 'macosx'
      config.build_settings["ONLY_ACTIVE_ARCH"] = 'YES'
    end
  end

  project.save

end


