#!/usr/bin/env ruby

require 'nokogiri'
require 'open-uri'
require 'json'
require 'uri'

class GoogleScholarFetcher
  def initialize(scholar_id)
    @scholar_id = scholar_id
    @base_url = "https://scholar.google.com"
    @profile_url = "#{@base_url}/citations?user=#{@scholar_id}&hl=en"
  end

  def fetch_publications
    puts "Fetching publications from Google Scholar..."
    puts "Profile URL: #{@profile_url}"
    
    begin
      # Add a user agent to avoid being blocked
      user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
      
      doc = Nokogiri::HTML(URI.open(@profile_url, "User-Agent" => user_agent))
      
      publications = []
      
      # Look for publication entries
      doc.css('.gs_r.gs_or.gs_scl').each do |publication|
        title_element = publication.css('.gs_rt a').first
        next unless title_element
        
        title = title_element.text.strip
        url = title_element['href']
        
        # Extract authors
        authors_element = publication.css('.gs_a').first
        authors = authors_element ? authors_element.text.split('-').first.strip : ""
        
        # Extract year
        year_match = authors_element&.text&.match(/(\d{4})/)
        year = year_match ? year_match[1] : ""
        
        # Extract venue/journal
        venue_element = publication.css('.gs_a').first
        venue = venue_element ? venue_element.text.split('-')[1]&.strip : ""
        
        # Extract citations
        citations_element = publication.css('.gs_fl a').find { |a| a.text.include?('Cited by') }
        citations = citations_element ? citations_element.text.match(/(\d+)/)&.[](1) : "0"
        
        publications << {
          title: title,
          authors: authors,
          year: year,
          venue: venue,
          url: url,
          citations: citations
        }
      end
      
      publications
    rescue => e
      puts "Error fetching publications: #{e.message}"
      []
    end
  end

  def generate_bibtex(publications)
    bibtex_entries = []
    
    publications.each_with_index do |pub, index|
      # Generate a unique key for the publication
      key = generate_key(pub[:title], pub[:authors], pub[:year])
      
      entry = <<~BIBTEX
        @article{#{key},
          title={#{pub[:title]}},
          author={#{pub[:authors]}},
          year={#{pub[:year]}},
          journal={#{pub[:venue]}},
          url={#{pub[:url]}},
          selected={true}
        }
        
      BIBTEX
      
      bibtex_entries << entry
    end
    
    bibtex_entries.join("\n")
  end

  def generate_key(title, authors, year)
    # Generate a simple key from title and first author
    first_author = authors.split(',').first.strip.split(' ').last.downcase
    title_words = title.split(' ').first(3).map(&:downcase).join('_')
    "#{first_author}#{year}#{title_words}".gsub(/[^a-zA-Z0-9_]/, '')
  end

  def save_to_file(bibtex_content, filename = "_bibliography/new_papers.bib")
    File.write(filename, bibtex_content)
    puts "Saved #{filename}"
  end
end

# Main execution
if __FILE__ == $0
  scholar_id = "_xugfIEAAAAJ"
  fetcher = GoogleScholarFetcher.new(scholar_id)
  
  publications = fetcher.fetch_publications
  
  if publications.empty?
    puts "No publications found or error occurred."
  else
    puts "Found #{publications.length} publications:"
    publications.each_with_index do |pub, index|
      puts "#{index + 1}. #{pub[:title]} (#{pub[:year]})"
    end
    
    bibtex_content = fetcher.generate_bibtex(publications)
    fetcher.save_to_file(bibtex_content)
    
    puts "\nBibTeX entries have been saved to _bibliography/new_papers.bib"
    puts "You can review and merge them with your existing papers.bib file."
  end
end 