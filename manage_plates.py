#!/usr/bin/env python3
"""
Plate Management Tool
CLI tool for managing known plates database
"""

import sys
import argparse
from pathlib import Path
from database_manager import DatabaseManager
from tabulate import tabulate
from datetime import datetime

class PlateManager:
    def __init__(self, db_path="database/security.db"):
        self.db = DatabaseManager(db_path)
    
    def add_plate(self, plate, owner, vehicle_type="", alert_type="known", notes=""):
        """Add a new plate"""
        success = self.db.add_known_plate(plate, owner, vehicle_type, alert_type, notes)
        if success:
            print(f"✓ Added plate {plate} ({alert_type})")
        else:
            print(f"✗ Failed to add plate {plate}")
    
    def remove_plate(self, plate):
        """Remove a plate"""
        success = self.db.remove_plate(plate)
        if success:
            print(f"✓ Removed plate {plate}")
        else:
            print(f"✗ Failed to remove plate {plate}")
    
    def list_plates(self, filter_type=None):
        """List all known plates"""
        plates = self.db.get_all_known_plates()
        
        if filter_type:
            plates = [p for p in plates if p['alert_type'] == filter_type]
        
        if not plates:
            print("No plates found")
            return
        
        # Prepare table data
        table_data = []
        for plate in plates:
            table_data.append([
                plate['plate_number'],
                plate['owner_name'],
                plate['vehicle_type'] or '-',
                plate['alert_type'],
                plate['last_seen'][:16] if plate['last_seen'] else 'Never',
                plate['notes'] or '-'
            ])
        
        headers = ['Plate', 'Owner', 'Vehicle', 'Type', 'Last Seen', 'Notes']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"\nTotal: {len(plates)} plates")
    
    def search_plate(self, plate):
        """Search for a specific plate"""
        info = self.db.check_plate(plate)
        
        if not info:
            print(f"Plate {plate} not found")
            return
        
        print(f"\n=== Plate Information ===")
        print(f"Plate Number:  {info['plate_number']}")
        print(f"Owner:         {info['owner_name']}")
        print(f"Vehicle Type:  {info['vehicle_type'] or 'N/A'}")
        print(f"Alert Type:    {info['alert_type']}")
        print(f"Added:         {info['added_date'][:16]}")
        print(f"Last Seen:     {info['last_seen'][:16] if info['last_seen'] else 'Never'}")
        print(f"Notes:         {info['notes'] or 'None'}")
        
        # Get history
        history = self.db.get_plate_history(plate, limit=10)
        if history:
            print(f"\n=== Recent Detections (last 10) ===")
            table_data = []
            for event in history:
                table_data.append([
                    event['timestamp'][:16],
                    f"{event['confidence']:.2f}" if event['confidence'] else '-',
                    event['image_path'].split('/')[-1] if event['image_path'] else '-'
                ])
            print(tabulate(table_data, headers=['Timestamp', 'Confidence', 'Image'], 
                          tablefmt='grid'))
    
    def show_statistics(self, hours=24):
        """Show system statistics"""
        stats = self.db.get_statistics(hours)
        
        print(f"\n=== Statistics (last {hours} hours) ===")
        print(f"Total Events:        {stats.get('total_events', 0)}")
        print(f"Unique Plates:       {stats.get('unique_plates', 0)}")
        print(f"People Detections:   {stats.get('people_detections', 0)}")
        print(f"Blacklist Alerts:    {stats.get('blacklist_alerts', 0)}")
        
        # Plate breakdown
        plates = self.db.get_all_known_plates()
        known_count = len([p for p in plates if p['alert_type'] == 'known'])
        blacklist_count = len([p for p in plates if p['alert_type'] == 'blacklist'])
        
        print(f"\n=== Database Summary ===")
        print(f"Total Known Plates:  {len(plates)}")
        print(f"  - Known:           {known_count}")
        print(f"  - Blacklisted:     {blacklist_count}")
    
    def recent_events(self, hours=24, limit=20):
        """Show recent events"""
        events = self.db.get_recent_events(hours=hours, limit=limit)
        
        if not events:
            print(f"No events in the last {hours} hours")
            return
        
        print(f"\n=== Recent Events (last {hours} hours) ===")
        table_data = []
        for event in events:
            table_data.append([
                event['timestamp'][:16],
                event['event_type'],
                event['plate_number'] or '-',
                f"{event['confidence']:.2f}" if event['confidence'] else '-',
                '✓' if event['alerted'] else '✗'
            ])
        
        print(tabulate(table_data, 
                      headers=['Timestamp', 'Type', 'Plate', 'Conf', 'Alerted'],
                      tablefmt='grid'))
        print(f"\nShowing {len(events)} events")
    
    def import_from_csv(self, csv_file):
        """Import plates from CSV file"""
        try:
            import csv
            count = 0
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.db.add_known_plate(
                        row['plate'],
                        row['owner'],
                        row.get('vehicle_type', ''),
                        row.get('alert_type', 'known'),
                        row.get('notes', '')
                    )
                    count += 1
            print(f"✓ Imported {count} plates from {csv_file}")
        except Exception as e:
            print(f"✗ Import failed: {e}")
    
    def export_to_csv(self, csv_file):
        """Export plates to CSV file"""
        try:
            import csv
            plates = self.db.get_all_known_plates()
            
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['plate', 'owner', 'vehicle_type', 'alert_type', 
                             'added_date', 'last_seen', 'notes']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for plate in plates:
                    writer.writerow({
                        'plate': plate['plate_number'],
                        'owner': plate['owner_name'],
                        'vehicle_type': plate['vehicle_type'],
                        'alert_type': plate['alert_type'],
                        'added_date': plate['added_date'],
                        'last_seen': plate['last_seen'] or '',
                        'notes': plate['notes']
                    })
            
            print(f"✓ Exported {len(plates)} plates to {csv_file}")
        except Exception as e:
            print(f"✗ Export failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Manage security camera plate database')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add plate
    add_parser = subparsers.add_parser('add', help='Add a new plate')
    add_parser.add_argument('plate', help='Plate number')
    add_parser.add_argument('owner', help='Owner name')
    add_parser.add_argument('--vehicle', default='', help='Vehicle type')
    add_parser.add_argument('--type', default='known', 
                           choices=['known', 'blacklist', 'whitelist'],
                           help='Alert type')
    add_parser.add_argument('--notes', default='', help='Additional notes')
    
    # Remove plate
    remove_parser = subparsers.add_parser('remove', help='Remove a plate')
    remove_parser.add_argument('plate', help='Plate number')
    
    # List plates
    list_parser = subparsers.add_parser('list', help='List all plates')
    list_parser.add_argument('--type', choices=['known', 'blacklist', 'whitelist'],
                            help='Filter by type')
    
    # Search plate
    search_parser = subparsers.add_parser('search', help='Search for a plate')
    search_parser.add_argument('plate', help='Plate number')
    
    # Statistics
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--hours', type=int, default=24, 
                             help='Time period in hours')
    
    # Recent events
    events_parser = subparsers.add_parser('events', help='Show recent events')
    events_parser.add_argument('--hours', type=int, default=24,
                              help='Time period in hours')
    events_parser.add_argument('--limit', type=int, default=20,
                              help='Max number of events')
    
    # Import/Export
    import_parser = subparsers.add_parser('import', help='Import from CSV')
    import_parser.add_argument('file', help='CSV file path')
    
    export_parser = subparsers.add_parser('export', help='Export to CSV')
    export_parser.add_argument('file', help='CSV file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = PlateManager()
    
    if args.command == 'add':
        manager.add_plate(args.plate, args.owner, args.vehicle, args.type, args.notes)
    elif args.command == 'remove':
        manager.remove_plate(args.plate)
    elif args.command == 'list':
        manager.list_plates(args.type)
    elif args.command == 'search':
        manager.search_plate(args.plate)
    elif args.command == 'stats':
        manager.show_statistics(args.hours)
    elif args.command == 'events':
        manager.recent_events(args.hours, args.limit)
    elif args.command == 'import':
        manager.import_from_csv(args.file)
    elif args.command == 'export':
        manager.export_to_csv(args.file)

if __name__ == "__main__":
    main()
