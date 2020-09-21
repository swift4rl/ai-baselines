#!/usr/bin/env ruby

require 'xcodeproj'

task :dependencies do

  system "swift package update"
  system "swift package generate-xcodeproj"

  project = Xcodeproj::Project.open('ai-baselines.xcodeproj')

  project.targets.each do |target|
    project.build_configurations.each { |config|
      config.build_settings["MACOSX_DEPLOYMENT_TARGET"] = 10.15
      config.build_settings["SUPPORTED_PLATFORMS"] = 'macOS'
    }

    target.build_configurations.each do |config|
      config.build_settings["MACOSX_DEPLOYMENT_TARGET"] = 10.15
    end
  end

  project.save

end
