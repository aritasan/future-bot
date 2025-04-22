"""
Script to analyze closed orders from Binance futures account.
"""
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.config import load_config
from src.services.binance_service import BinanceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderAnalyzer:
    def __init__(self, binance_service: BinanceService):
        self.binance_service = binance_service
        self.closed_orders = []
        
    async def fetch_closed_orders(self, days: int = 30) -> List[Dict]:
        """Fetch closed orders from the last N days."""
        try:
            # Calculate start time
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Get all markets
            markets = await self.binance_service.get_markets()
            if not markets:
                logger.error("Failed to get markets")
                return []
                
            # Fetch orders for each market
            all_orders = []
            for symbol in markets:
                try:
                    orders = await self.binance_service.exchange.fetch_closed_orders(
                        symbol=symbol,
                        since=start_time,
                        limit=1000
                    )
                    all_orders.extend(orders)
                    logger.info(f"Fetched {len(orders)} closed orders for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching orders for {symbol}: {str(e)}")
                    continue
                    
            return all_orders
        except Exception as e:
            logger.error(f"Error fetching closed orders: {str(e)}")
            return []
            
    async def analyze_orders(self) -> Dict:
        """Analyze closed orders and generate statistics."""
        try:
            if not self.closed_orders:
                logger.warning("No closed orders to analyze")
                return {}
                
            total_orders = len(self.closed_orders)
            winning_orders = 0
            total_pnl = 0.0
            symbol_stats = {}
            
            for order in self.closed_orders:
                # Calculate realized PnL from order data
                realized_pnl = 0.0
                if 'info' in order and 'realizedPnl' in order['info']:
                    realized_pnl = float(order['info']['realizedPnl'])
                elif 'cost' in order and 'filled' in order:
                    # If realizedPnl is not available, calculate from cost and filled amount
                    cost = float(order.get('cost', 0))
                    filled = float(order.get('filled', 0))
                    price = float(order.get('price', 0))
                    if filled > 0 and price > 0:
                        realized_pnl = (price - cost) * filled
                
                # Update statistics
                if realized_pnl > 0:
                    winning_orders += 1
                total_pnl += realized_pnl
                
                # Update symbol-specific statistics
                symbol = order.get('symbol', 'UNKNOWN')
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'total_orders': 0,
                        'winning_orders': 0,
                        'total_pnl': 0.0
                    }
                symbol_stats[symbol]['total_orders'] += 1
                if realized_pnl > 0:
                    symbol_stats[symbol]['winning_orders'] += 1
                symbol_stats[symbol]['total_pnl'] += realized_pnl
                
                # Classify order type
                order_type = self._classify_order(order)
                order['order_type'] = order_type
                
            # Calculate overall statistics
            win_rate = (winning_orders / total_orders) * 100 if total_orders > 0 else 0
            avg_pnl = total_pnl / total_orders if total_orders > 0 else 0
            
            return {
                'total_orders': total_orders,
                'winning_orders': winning_orders,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'symbol_stats': symbol_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing orders: {str(e)}")
            return {}
            
    def _classify_order(self, order: pd.Series) -> str:
        """Classify order based on its properties."""
        try:
            # Check if it's a stop loss
            if order.get('stopPrice') and order.get('price') == order.get('stopPrice'):
                return 'stop_loss'
                
            # Check if it's a take profit
            if order.get('takeProfitPrice') and order.get('price') == order.get('takeProfitPrice'):
                return 'take_profit'
                
            # Check if it's a market order
            if order.get('type') == 'market':
                return 'market'
                
            # Default to limit order
            return 'limit'
        except Exception as e:
            logger.error(f"Error classifying order: {str(e)}")
            return 'unknown'
            
    def generate_report(self, stats: Dict) -> str:
        """Generate a human-readable report from statistics."""
        try:
            report = []
            report.append("📊 Trading Performance Report")
            report.append("=" * 50)
            
            # Overall statistics
            report.append("\nOverall Statistics:")
            report.append(f"Total Orders: {stats.get('total_orders', 0)}")
            report.append(f"Winning Orders: {stats.get('winning_orders', 0)}")
            report.append(f"Win Rate: {stats.get('win_rate', 0):.2f}%")
            report.append(f"Total PnL: {stats.get('total_pnl', 0):.2f} USDT")
            report.append(f"Average PnL: {stats.get('avg_pnl', 0):.2f} USDT")
            
            # Order type breakdown
            report.append("\nOrder Type Breakdown:")
            for order_type, count in stats.get('order_types', {}).items():
                report.append(f"{order_type}: {count} ({count/stats['total_orders']*100:.2f}%)")
                
            # Symbol-specific statistics
            report.append("\nSymbol-Specific Statistics:")
            for symbol, symbol_data in stats.get('symbol_stats', {}).items():
                report.append(f"\n{symbol}:")
                report.append(f"  Total Orders: {symbol_data['total_orders']}")
                report.append(f"  Total PnL: {symbol_data['total_pnl']:.2f} USDT")
            
            return "\n".join(report)
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return "Error generating report"
            
    def plot_performance(self, orders: List[Dict], output_file: str = "trading_performance.png"):
        """Generate performance visualization plots."""
        try:
            if not orders:
                return
                
            # Convert to DataFrame
            df = pd.DataFrame(orders)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['pnl'] = df['realized_pnl'].astype(float)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Cumulative PnL over time
            df['cumulative_pnl'] = df['pnl'].cumsum()
            sns.lineplot(data=df, x='timestamp', y='cumulative_pnl', ax=axes[0, 0])
            axes[0, 0].set_title('Cumulative PnL Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Cumulative PnL (USDT)')
            
            # 2. PnL distribution
            sns.histplot(data=df, x='pnl', bins=50, ax=axes[0, 1])
            axes[0, 1].set_title('PnL Distribution')
            axes[0, 1].set_xlabel('PnL (USDT)')
            axes[0, 1].set_ylabel('Count')
            
            # 3. Win rate by symbol
            symbol_stats = df.groupby('symbol').agg({
                'pnl': lambda x: (x > 0).mean() * 100
            }).reset_index()
            sns.barplot(data=symbol_stats, x='symbol', y='pnl', ax=axes[1, 0])
            axes[1, 0].set_title('Win Rate by Symbol')
            axes[1, 0].set_xlabel('Symbol')
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Average PnL by symbol
            symbol_pnl = df.groupby('symbol')['pnl'].mean().reset_index()
            sns.barplot(data=symbol_pnl, x='symbol', y='pnl', ax=axes[1, 1])
            axes[1, 1].set_title('Average PnL by Symbol')
            axes[1, 1].set_xlabel('Symbol')
            axes[1, 1].set_ylabel('Average PnL (USDT)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Performance plots saved to {output_file}")
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            
    def export_to_csv(self, orders: List[Dict], output_file: str = "closed_orders.csv"):
        """Export order data to CSV file."""
        try:
            if not orders:
                logger.warning("No orders to export")
                return False
                
            # Convert to DataFrame
            df = pd.DataFrame(orders)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            # Extract realized PnL
            if 'info' in df.columns:
                # Try to extract realizedPnl from info column
                df['realized_pnl'] = df['info'].apply(
                    lambda x: float(x.get('realizedPnl', 0)) if isinstance(x, dict) and 'realizedPnl' in x else 0
                )
            elif 'realized_pnl' not in df.columns:
                # Calculate PnL if not available
                df['realized_pnl'] = 0
                if 'cost' in df.columns and 'filled' in df.columns and 'price' in df.columns:
                    df['realized_pnl'] = (df['price'] - df['cost']) * df['filled']
                    
            # Select and rename columns for export
            export_columns = [
                'symbol', 'id', 'timestamp', 'datetime', 'type', 'side', 
                'price', 'amount', 'filled', 'cost', 'fee', 'realized_pnl',
                'status', 'order_type'
            ]
            
            # Filter columns that exist in the DataFrame
            available_columns = [col for col in export_columns if col in df.columns]
            export_df = df[available_columns]
            
            # Save to CSV
            export_df.to_csv(output_file, index=False)
            logger.info(f"Order data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False

async def main():
    try:
        # Set event loop policy for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        # Load configuration
        config = load_config()
        
        # Initialize Binance service
        binance_service = BinanceService(config)
        if not await binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
            
        # Create analyzer
        analyzer = OrderAnalyzer(binance_service)
        
        # Fetch closed orders (last 30 days)
        logger.info("Fetching closed orders...")
        orders = await analyzer.fetch_closed_orders(days=30)
        
        if not orders:
            logger.error("No closed orders found")
            return
            
        # Store orders in analyzer
        analyzer.closed_orders = orders
            
        # Analyze orders
        logger.info("Analyzing orders...")
        stats = await analyzer.analyze_orders()
        
        # Generate report
        report = analyzer.generate_report(stats)
        print("\n" + report)
        
        # Export to CSV
        logger.info("Exporting order data to CSV...")
        analyzer.export_to_csv(orders)
        
        # Generate performance plots
        logger.info("Generating performance plots...")
        analyzer.plot_performance(orders)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        # Cleanup
        if 'binance_service' in locals():
            await binance_service.close()

if __name__ == "__main__":
    try:
        # Set event loop policy for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1) 