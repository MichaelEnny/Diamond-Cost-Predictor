"""
Comprehensive API Testing Suite for Diamond Price Predictor
Tests all endpoints with various scenarios
"""
import requests
import json
import time
import threading
from typing import Dict, Any, List
import statistics


class APITester:
    """Comprehensive API testing class"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Test results tracking
        self.test_results = []
        self.performance_metrics = []
    
    def log_test(self, test_name: str, success: bool, details: str = "", response_time_ms: float = 0):
        """Log test result"""
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'response_time_ms': response_time_ms
        })
        
        if response_time_ms > 0:
            self.performance_metrics.append(response_time_ms)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
    
    def test_health_check(self):
        """Test health check endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/v1/health")
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and data['data']['api_status'] in ['healthy', 'degraded']:
                    self.log_test(
                        "Health Check", 
                        True, 
                        f"API Status: {data['data']['api_status']}", 
                        response_time
                    )
                else:
                    self.log_test("Health Check", False, "Invalid health response format")
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
    
    def test_api_info(self):
        """Test API info endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/v1/info")
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and 'endpoints' in data['data']:
                    endpoint_count = len(data['data']['endpoints'])
                    self.log_test(
                        "API Info", 
                        True, 
                        f"Retrieved info with {endpoint_count} endpoints", 
                        response_time
                    )
                else:
                    self.log_test("API Info", False, "Invalid info response format")
            else:
                self.log_test("API Info", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("API Info", False, f"Exception: {str(e)}")
    
    def test_single_prediction_valid(self):
        """Test single prediction with valid data"""
        try:
            diamond_data = {
                "carat": 1.0,
                "cut": "Ideal",
                "color": "E",
                "clarity": "VS1",
                "depth": 61.5,
                "table": 55.0,
                "x": 6.0,
                "y": 6.0,
                "z": 3.7
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/v1/predict", 
                json=diamond_data
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and 'predicted_price' in data['data']:
                    price = data['data']['predicted_price']
                    inference_time = data['data']['inference_time_ms']
                    
                    if 100 <= price <= 50000:  # Reasonable price range
                        self.log_test(
                            "Single Prediction (Valid)", 
                            True, 
                            f"Price: ${price:,.2f}, Inference: {inference_time}ms", 
                            response_time
                        )
                    else:
                        self.log_test(
                            "Single Prediction (Valid)", 
                            False, 
                            f"Price out of reasonable range: ${price:,.2f}"
                        )
                else:
                    self.log_test("Single Prediction (Valid)", False, "Invalid prediction response")
            else:
                self.log_test("Single Prediction (Valid)", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Single Prediction (Valid)", False, f"Exception: {str(e)}")
    
    def test_single_prediction_invalid(self):
        """Test single prediction with invalid data"""
        test_cases = [
            {
                "name": "Missing Fields",
                "data": {"carat": 1.0},
                "expected_status": 400
            },
            {
                "name": "Invalid Cut",
                "data": {
                    "carat": 1.0, "cut": "Invalid", "color": "E", "clarity": "VS1",
                    "depth": 61.5, "table": 55.0, "x": 6.0, "y": 6.0, "z": 3.7
                },
                "expected_status": 400
            },
            {
                "name": "Negative Carat",
                "data": {
                    "carat": -1.0, "cut": "Ideal", "color": "E", "clarity": "VS1",
                    "depth": 61.5, "table": 55.0, "x": 6.0, "y": 6.0, "z": 3.7
                },
                "expected_status": 400
            },
            {
                "name": "Zero Dimensions",
                "data": {
                    "carat": 1.0, "cut": "Ideal", "color": "E", "clarity": "VS1",
                    "depth": 61.5, "table": 55.0, "x": 0, "y": 0, "z": 0
                },
                "expected_status": 400
            }
        ]
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/predict", 
                    json=test_case["data"]
                )
                
                if response.status_code == test_case["expected_status"]:
                    self.log_test(
                        f"Single Prediction Invalid ({test_case['name']})", 
                        True, 
                        f"Correctly rejected with status {response.status_code}"
                    )
                else:
                    self.log_test(
                        f"Single Prediction Invalid ({test_case['name']})", 
                        False, 
                        f"Expected {test_case['expected_status']}, got {response.status_code}"
                    )
                    
            except Exception as e:
                self.log_test(
                    f"Single Prediction Invalid ({test_case['name']})", 
                    False, 
                    f"Exception: {str(e)}"
                )
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        try:
            diamonds_data = {
                "diamonds": [
                    {
                        "carat": 0.5, "cut": "Good", "color": "G", "clarity": "SI1",
                        "depth": 62.0, "table": 56.0, "x": 5.0, "y": 5.0, "z": 3.1
                    },
                    {
                        "carat": 1.5, "cut": "Premium", "color": "F", "clarity": "VVS2",
                        "depth": 61.0, "table": 57.0, "x": 7.0, "y": 7.0, "z": 4.3
                    },
                    {
                        "carat": 2.0, "cut": "Ideal", "color": "D", "clarity": "IF",
                        "depth": 60.5, "table": 55.5, "x": 8.0, "y": 8.0, "z": 4.8
                    }
                ]
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/v1/predict/batch", 
                json=diamonds_data
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and 'predictions' in data['data']:
                    predictions = data['data']['predictions']
                    batch_summary = data['data']['batch_summary']
                    
                    if len(predictions) == 3 and batch_summary['successful_predictions'] == 3:
                        total_value = sum(p['predicted_price'] for p in predictions)
                        self.log_test(
                            "Batch Prediction", 
                            True, 
                            f"3 diamonds, Total value: ${total_value:,.2f}", 
                            response_time
                        )
                    else:
                        self.log_test("Batch Prediction", False, "Incomplete batch results")
                else:
                    self.log_test("Batch Prediction", False, "Invalid batch response format")
            else:
                self.log_test("Batch Prediction", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Batch Prediction", False, f"Exception: {str(e)}")
    
    def test_validation_endpoint(self):
        """Test validation endpoint"""
        try:
            valid_data = {
                "carat": 1.0, "cut": "Ideal", "color": "E", "clarity": "VS1",
                "depth": 61.5, "table": 55.0, "x": 6.0, "y": 6.0, "z": 3.7
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/v1/validate", 
                json=valid_data
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and data['data']['is_valid']:
                    self.log_test(
                        "Validation Endpoint", 
                        True, 
                        "Valid data correctly accepted", 
                        response_time
                    )
                else:
                    self.log_test("Validation Endpoint", False, "Valid data rejected")
            else:
                self.log_test("Validation Endpoint", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Validation Endpoint", False, f"Exception: {str(e)}")
    
    def test_model_info(self):
        """Test model info endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/v1/model/info")
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and 'model_type' in data['data']:
                    model_type = data['data']['model_type']
                    self.log_test(
                        "Model Info", 
                        True, 
                        f"Model type: {model_type}", 
                        response_time
                    )
                else:
                    self.log_test("Model Info", False, "Invalid model info response")
            else:
                self.log_test("Model Info", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Model Info", False, f"Exception: {str(e)}")
    
    def test_performance_load(self, num_requests: int = 50):
        """Test API performance under load"""
        try:
            diamond_data = {
                "caret": 1.0, "cut": "Ideal", "color": "E", "clarity": "VS1",
                "depth": 61.5, "table": 55.0, "x": 6.0, "y": 6.0, "z": 3.7
            }
            
            response_times = []
            successful_requests = 0
            failed_requests = 0
            
            def make_request():
                nonlocal successful_requests, failed_requests
                try:
                    start_time = time.time()
                    response = self.session.post(
                        f"{self.base_url}/api/v1/predict", 
                        json=diamond_data
                    )
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        successful_requests += 1
                        response_times.append(response_time)
                    else:
                        failed_requests += 1
                        
                except Exception:
                    failed_requests += 1
            
            # Execute concurrent requests
            threads = []
            start_time = time.time()
            
            for _ in range(num_requests):
                thread = threading.Thread(target=make_request)
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            if successful_requests > 0:
                avg_response_time = statistics.mean(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                throughput = successful_requests / total_time
                
                # Performance thresholds
                performance_ok = (
                    avg_response_time <= 200 and  # Target: <200ms
                    throughput >= 10 and  # Target: >10 requests/second
                    successful_requests / num_requests >= 0.95  # Target: >95% success rate
                )
                
                details = (f"Avg: {avg_response_time:.1f}ms, "
                          f"Max: {max_response_time:.1f}ms, "
                          f"Throughput: {throughput:.1f} req/s, "
                          f"Success: {successful_requests}/{num_requests}")
                
                self.log_test("Performance Load Test", performance_ok, details)
            else:
                self.log_test("Performance Load Test", False, "No successful requests")
                
        except Exception as e:
            self.log_test("Performance Load Test", False, f"Exception: {str(e)}")
    
    def test_error_handling(self):
        """Test various error conditions"""
        error_tests = [
            {
                "name": "Non-JSON Request",
                "url": f"{self.base_url}/api/v1/predict",
                "data": "invalid json",
                "headers": {"Content-Type": "text/plain"},
                "expected_status": 400
            },
            {
                "name": "Empty JSON",
                "url": f"{self.base_url}/api/v1/predict",
                "data": {},
                "headers": {"Content-Type": "application/json"},
                "expected_status": 400
            },
            {
                "name": "Invalid Endpoint",
                "url": f"{self.base_url}/api/v1/nonexistent",
                "data": None,
                "headers": None,
                "expected_status": 404
            }
        ]
        
        for test in error_tests:
            try:
                if test["data"] is None:
                    response = self.session.get(test["url"])
                else:
                    response = requests.post(
                        test["url"], 
                        data=json.dumps(test["data"]) if isinstance(test["data"], dict) else test["data"],
                        headers=test["headers"] or {}
                    )
                
                if response.status_code == test["expected_status"]:
                    self.log_test(
                        f"Error Handling ({test['name']})", 
                        True, 
                        f"Correctly returned status {response.status_code}"
                    )
                else:
                    self.log_test(
                        f"Error Handling ({test['name']})", 
                        False, 
                        f"Expected {test['expected_status']}, got {response.status_code}"
                    )
                    
            except Exception as e:
                self.log_test(
                    f"Error Handling ({test['name']})", 
                    False, 
                    f"Exception: {str(e)}"
                )
    
    def run_all_tests(self):
        """Run all test suites"""
        print("="*80)
        print("🧪 DIAMOND PRICE PREDICTOR API TEST SUITE")
        print("="*80)
        
        # Basic endpoint tests
        print("\n📡 Testing Basic Endpoints...")
        self.test_health_check()
        self.test_api_info()
        self.test_model_info()
        
        # Functionality tests  
        print("\n🔮 Testing Prediction Functionality...")
        self.test_single_prediction_valid()
        self.test_single_prediction_invalid()
        self.test_batch_prediction()
        self.test_validation_endpoint()
        
        # Error handling tests
        print("\n🚨 Testing Error Handling...")
        self.test_error_handling()
        
        # Performance tests
        print("\n⚡ Testing Performance...")
        self.test_performance_load(20)  # Reduced for faster testing
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['success']])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.performance_metrics:
            avg_response_time = statistics.mean(self.performance_metrics)
            max_response_time = max(self.performance_metrics)
            min_response_time = min(self.performance_metrics)
            
            print(f"\n⚡ Performance Metrics:")
            print(f"Average Response Time: {avg_response_time:.2f}ms")
            print(f"Maximum Response Time: {max_response_time:.2f}ms")
            print(f"Minimum Response Time: {min_response_time:.2f}ms")
            print(f"Target Met (<200ms): {'✅ YES' if avg_response_time < 200 else '❌ NO'}")
        
        # Show failed tests
        failed_test_details = [t for t in self.test_results if not t['success']]
        if failed_test_details:
            print(f"\n❌ Failed Tests Details:")
            for test in failed_test_details:
                print(f"  • {test['test']}: {test['details']}")
        
        print("\n" + "="*80)
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! API IS READY FOR PRODUCTION!")
        else:
            print("⚠️  SOME TESTS FAILED. PLEASE REVIEW AND FIX ISSUES BEFORE PRODUCTION.")
        
        print("="*80)


def main():
    """Main testing function"""
    print("Starting API testing...")
    print("Make sure the API server is running on http://localhost:5000")
    
    # Check if API is accessible
    try:
        response = requests.get("http://localhost:5000/api/v1/health", timeout=5)
        if response.status_code in [200, 503]:  # 503 is acceptable for degraded health
            print("✅ API server detected")
        else:
            print(f"❌ API server returned status {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Please start the server first:")
        print("   python app.py")
        return
    except Exception as e:
        print(f"❌ Error connecting to API: {str(e)}")
        return
    
    # Run tests
    tester = APITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()