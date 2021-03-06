# encoding: utf-8
#!/usr/bin/env ruby

require 'optparse'

def print_sysinfo(outfile, argvstr, cc='#')
  print(argvstr)
  date = `date`.chomp
  pwd  = `pwd`.chomp
  hostname = `hostname`.chomp
  user = `whoami`.chomp
 
  outfile.print("#{cc} Generated #{date} by #{user}\n#{cc} #{hostname}:#{pwd}\n#{cc} #{$0} #{argvstr}\n\n")
end

argvstr = ARGV.join(" ")

# This hash will hold all of the options 
# parsed from the command-line by 
# OptionParser. 
options = {}

optparse = OptionParser.new do |opts|
  # Set a banner, displayed at the top 
  # of the help screen. 
  opts.banner = "Usage: #{$0} [options] <in.cdl> <out.cdl>"
  # Define the options, and what they do 
  options[:force] = false
  opts.on( '-f', '--force', 'Force to overwrite output file' ) do
    options[:force] = true
  end
  options[:capval] = 0.0
  opts.on( '-c', '--cap value', Float, 'CPP -> MOMCAPS' ) do|f|
    options[:capval] = f
  end
  options[:bracket] = false
  opts.on( '-b', '--bracket', 'Bracket <> -> []' ) do
    options[:bracket] = true
  end
  options[:dummy] = false
  opts.on( '-d', '--dummy', 'Dummy cap and resistor removal' ) do
    options[:dummy] = true
  end
  options[:logfile] = nil
  opts.on( '-l', '--logfile FILE', 'Write log to FILE' ) do|file|
    options[:logfile] = file
  end
  # This displays the help screen, all programs are 
  # assumed to have this option. 
  opts.on( '-h', '--help', 'Display this screen' ) do
    puts opts
    exit
  end
end

# Parse the command-line. The 'parse' method simply parses 
# ARGV, while the 'parse!' method parses ARGV and removes 
# any options found there, as well as any parameters for 
# the options. What's left is the list of files to resize.


begin optparse.parse!(ARGV)
 rescue OptionParser::InvalidOption => e
 puts e
 puts optparse
 exit
end

puts "Force to overwrite output file" if options[:force]
puts "CPP -> MOMCAPS #{options[:capval]}" if options[:capval]
puts "Bracket <> -> []" if options[:bracket]
puts "Dummy cap & resistor removal" if options[:dummy]
puts "Logging to file #{options[:logfile]}" if options[:logfile]


outfile = File.open(options[:logfile], 'w')
print_sysinfo(outfile, argvstr, '//')
