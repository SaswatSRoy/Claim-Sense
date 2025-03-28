package android.saswat.dashcamapplication

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Application
import android.content.Context
import android.util.Log
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sign

private const val TAG = "DashCamViewModel"
// Constants for sensor data indices
private const val ACC_X = 0
private const val ACC_Y = 1
private const val ACC_Z = 2
private const val GYRO_X = 3
private const val GYRO_Y = 4
private const val GYRO_Z = 5
// Constants for risk calculation
private const val DEFAULT_RISK_SCORE = 50f
private const val SENSORY_WEIGHT = 0.6f
private const val IMAGE_WEIGHT = 0.4f
// Constants for risk smoothing
private const val RISK_SMOOTHING_FACTOR = 0.2f
private const val RISK_HISTORY_SIZE = 5
private const val MAX_RISK_CHANGE_THRESHOLD = 10f
// Camera resolution
private const val CAMERA_RESOLUTION_WIDTH = 320
private const val CAMERA_RESOLUTION_HEIGHT = 320
// Auto sign out constants
private const val AUTO_SIGNOUT_DELAY_MS = 30 * 60 * 1000L // 30 minutes
private const val RISK_STORAGE_INTERVAL_MS = 10 * 60 * 1000L // 10 minutes

data class AuthState(
    val isAuthenticated: Boolean = false,
    val error: String? = null,
    val userId: String? = null,
    val isLoading: Boolean = false
)

data class ModelState(
    val isLoaded: Boolean = false,
    val error: String? = null
)

class DashCamViewModel(application: Application) : AndroidViewModel(application) {
    private val context = application.applicationContext
    private val auth = FirebaseAuth.getInstance()
    private val db = FirebaseFirestore.getInstance()
    
    private val _authState = MutableStateFlow(AuthState())
    val authState: StateFlow<AuthState> = _authState
    
    private val _modelState = MutableStateFlow(ModelState())
    val modelState: StateFlow<ModelState> = _modelState
    
    private val _riskScore = MutableStateFlow(DEFAULT_RISK_SCORE)
    val riskScore: StateFlow<Float> = _riskScore
    
    private var sensorySession: OrtSession? = null
    private val sensorData = FloatArray(6) { 0f }
    private var lastSensorUpdate = 0L
    
    private val riskHistory = ArrayDeque<Float>(RISK_HISTORY_SIZE)
    private var lastCalculatedRisk = DEFAULT_RISK_SCORE
    
    private var autoSignoutJob: Job? = null
    private val _isSignedOut = MutableStateFlow(false)
    val isSignedOut: StateFlow<Boolean> = _isSignedOut
    private var lastRiskStorageTime = 0L

    init {
        loadOnnxModel()
        checkCurrentUser()
        lastRiskStorageTime = System.currentTimeMillis()  // Initialize the last storage time
    }
    
    private fun checkCurrentUser() {
        // Instead, always start as logged out
        _authState.value = AuthState(isAuthenticated = false)
        
        // Or if you want to be explicit about signing out any cached user:
        if (auth.currentUser != null) {
            signOut()
        }
    }
    
    fun signInWithEmail(email: String, password: String) {
        // Set loading state to true when starting sign-in
        _authState.value = _authState.value.copy(
            isLoading = true, 
            error = null  // Clear any previous errors
        )
        
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val result = auth.signInWithEmailAndPassword(email, password).await()
                result.user?.let { user ->
                    _authState.value = AuthState(
                        isAuthenticated = true, 
                        userId = user.uid,
                        isLoading = false  // Reset loading state on success
                    )
                    _isSignedOut.value = false  // Reset sign out flag
                    startAutoSignoutTimer()  // Start auto sign-out timer
                } ?: run {
                    // Handle case where result.user is null but no exception was thrown
                    _authState.value = AuthState(
                        isAuthenticated = false,
                        error = "Sign in failed. Please try again.",
                        isLoading = false  // Reset loading state
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to sign in with email/password", e)
                _authState.value = AuthState(
                    isAuthenticated = false,
                    error = e.message ?: "Authentication failed",
                    isLoading = false  // Reset loading state on error
                )
            }
        }
    }
    
    // Helper function to get sensor data for debugging in UI
    fun getSensorDebugInfo(): String {
        val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
        val lastUpdateTime = if (lastSensorUpdate > 0) {
            dateFormat.format(java.util.Date(lastSensorUpdate))
        } else {
            "Never"
        }
        
        return "Sensor Data:\n" +
               "ACC_X: ${String.format(Locale.US, "%.3f", sensorData[ACC_X])}\n" +
               "ACC_Y: ${String.format(Locale.US, "%.3f", sensorData[ACC_Y])}\n" +
               "ACC_Z: ${String.format(Locale.US, "%.3f", sensorData[ACC_Z])}\n" +
               "GYRO_X: ${String.format(Locale.US, "%.3f", sensorData[GYRO_X])}\n" +
               "GYRO_Y: ${String.format(Locale.US, "%.3f", sensorData[GYRO_Y])}\n" +
               "GYRO_Z: ${String.format(Locale.US, "%.3f", sensorData[GYRO_Z])}\n" +
               "Last Update: $lastUpdateTime"
    }

    private fun loadOnnxModel() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val env = OrtEnvironment.getEnvironment()
                try {
                    context.assets.open("sensory_model.onnx").use { inputStream ->
                        val modelBytes = inputStream.readBytes()
                        sensorySession = env.createSession(modelBytes, OrtSession.SessionOptions())
                    }
                    _modelState.value = ModelState(isLoaded = true)
                } catch (e: java.io.FileNotFoundException) {
                    Log.e(TAG, "ONNX model file not found in assets", e)
                    _modelState.value = ModelState(isLoaded = false, error = "Model file not found: sensory_model.onnx")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load ONNX model", e)
                _modelState.value = ModelState(error = e.message)
            }
        }
    }
    
    fun updateSensorData(sensorType: Int, values: FloatArray) {
        when (sensorType) {
            android.hardware.Sensor.TYPE_ACCELEROMETER -> {
                sensorData[ACC_X] = values[0]
                sensorData[ACC_Y] = values[1]
                sensorData[ACC_Z] = values[2]
            }
            android.hardware.Sensor.TYPE_GYROSCOPE -> {
                sensorData[GYRO_X] = values[0]
                sensorData[GYRO_Y] = values[1]
                sensorData[GYRO_Z] = values[2]
            }
        }
        lastSensorUpdate = System.currentTimeMillis()
        resetAutoSignoutTimer()
    }
    
    fun calculateAndStoreRisk() {
        val rawRisk = calculateRisk()
        val smoothedRisk = smoothRiskScore(rawRisk)
        _riskScore.value = smoothedRisk
        
        // Only store risk at specified intervals
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastRiskStorageTime >= RISK_STORAGE_INTERVAL_MS) {
            Log.d(TAG, "INTERVAL CHECK: Time to store risk score (interval reached)")
            storeRiskInFirestore(smoothedRisk)
            lastRiskStorageTime = currentTime
        } else {
            val timeUntilNextStorage = RISK_STORAGE_INTERVAL_MS - (currentTime - lastRiskStorageTime)
            Log.d(TAG, "INTERVAL CHECK: Not storing risk yet. Next storage in ${timeUntilNextStorage/1000} seconds")
        }
        
        resetAutoSignoutTimer()
    }
    
    private fun smoothRiskScore(newRisk: Float): Float {
        // Add to history
        if (riskHistory.size >= RISK_HISTORY_SIZE) {
            riskHistory.removeFirst()
        }
        riskHistory.addLast(newRisk)
        
        // Method 1: Moving average
        val movingAverage = riskHistory.average().toFloat()
        
        // Method 2: Exponential smoothing
        val exponentialSmoothing = lastCalculatedRisk + RISK_SMOOTHING_FACTOR * (newRisk - lastCalculatedRisk)
        lastCalculatedRisk = exponentialSmoothing
        
        // Apply change threshold to limit sudden jumps
        val previousScore = _riskScore.value
        val scoreDifference = movingAverage - previousScore
        
        // If change is too large, limit it to the threshold
        return if (abs(scoreDifference) > MAX_RISK_CHANGE_THRESHOLD) {
            previousScore + sign(scoreDifference) * MAX_RISK_CHANGE_THRESHOLD
        } else {
            movingAverage
        }
    }
    
    private fun calculateRisk(): Float {
        try {
            if (sensorData.all { it == 0f }) {
                Log.w(TAG, "Cannot calculate risk: all sensor data is zero")
                return DEFAULT_RISK_SCORE
            }
            
            val session = sensorySession
            if (session == null) {
                Log.e(TAG, "Cannot calculate risk: model session is null")
                return DEFAULT_RISK_SCORE
            }
            
            val env = OrtEnvironment.getEnvironment()
            
            // Debug sensor input data
            Log.d(TAG, "MODEL INPUT - Sensor data being passed to model: " +
                  "ACC(${sensorData[ACC_X]}, ${sensorData[ACC_Y]}, ${sensorData[ACC_Z]}) " +
                  "GYRO(${sensorData[GYRO_X]}, ${sensorData[GYRO_Y]}, ${sensorData[GYRO_Z]})")
            
            // Create a tensor from the sensorData array
            try {
                val sensoryTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sensorData), longArrayOf(1, 6))
                Log.d(TAG, "Tensor created successfully, running model inference")
                
                val outputs = session.run(mapOf("float_input" to sensoryTensor))
                Log.d(TAG, "Model inference completed, processing outputs")
                
                val outputTensor = outputs.get(0)
                // Handle different output types
                when (val outputValue = outputTensor.value) {
                    is Array<*> -> {
                        if (outputValue.isNotEmpty()) {
                            when (val firstElement = outputValue[0]) {
                                is FloatArray -> {
                                    val sensoryProba = firstElement
                                    val sensoryScore = calculateSensoryRisk(sensoryProba)
                                    val imageScore = DEFAULT_RISK_SCORE // Placeholder for actual image score
                                    
                                    Log.d(TAG, "Model output probabilities: ${sensoryProba.joinToString()}")
                                    Log.d(TAG, "Calculated sensory risk score: $sensoryScore")
                                    
                                    // Weighted average of sensory and image scores
                                    return SENSORY_WEIGHT * sensoryScore + IMAGE_WEIGHT * imageScore
                                }
                                is DoubleArray -> {
                                    // Convert double array to float array if needed
                                    val floatArray = firstElement.map { it.toFloat() }.toFloatArray()
                                    val sensoryScore = calculateSensoryRisk(floatArray)
                                    val imageScore = DEFAULT_RISK_SCORE
                                    
                                    Log.d(TAG, "Model output probabilities (converted from Double): ${floatArray.joinToString()}")
                                    Log.d(TAG, "Calculated sensory risk score: $sensoryScore")
                                    
                                    return SENSORY_WEIGHT * sensoryScore + IMAGE_WEIGHT * imageScore
                                }
                                else -> {
                                    Log.e(TAG, "Unexpected first element type in array: ${firstElement?.javaClass?.name}")
                                }
                            }
                        }
                    }
                    is FloatArray -> {
                        // Handle when output is directly a FloatArray
                        val sensoryProba = outputValue
                        val sensoryScore = calculateSensoryRisk(sensoryProba)
                        val imageScore = DEFAULT_RISK_SCORE
                        
                        Log.d(TAG, "Model output probabilities: ${sensoryProba.joinToString()}")
                        Log.d(TAG, "Calculated sensory risk score: $sensoryScore")
                        
                        return SENSORY_WEIGHT * sensoryScore + IMAGE_WEIGHT * imageScore
                    }
                    is LongArray -> {
                        // Convert long array to float array if that's what's coming back
                        val sensoryProba = outputValue.map { it.toFloat() }.toFloatArray()
                        val sensoryScore = calculateSensoryRisk(sensoryProba)
                        val imageScore = DEFAULT_RISK_SCORE
                        
                        Log.d(TAG, "Model output probabilities (converted from Long): ${sensoryProba.joinToString()}")
                        Log.d(TAG, "Calculated sensory risk score: $sensoryScore")
                        
                        return SENSORY_WEIGHT * sensoryScore + IMAGE_WEIGHT * imageScore
                    }
                    else -> {
                        Log.e(TAG, "Unexpected model output format type: ${outputValue?.javaClass?.name}, value: $outputValue")
                        return DEFAULT_RISK_SCORE
                    }
                }
                
                Log.e(TAG, "Failed to process model output")
                return DEFAULT_RISK_SCORE
            } catch (e: Exception) {
                Log.e(TAG, "Error during model inference", e)
                return DEFAULT_RISK_SCORE
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating risk", e)
            return DEFAULT_RISK_SCORE
        }
    }
    
    private fun calculateSensoryRisk(proba: FloatArray): Float {
        val predClass = proba.indices.maxByOrNull { proba[it] } ?: 0
        val classProb = proba.maxOrNull() ?: 0f
        return when (predClass) {
            0 -> 80 + classProb * 20  // Aggressive
            1 -> classProb * 30       // Normal
            else -> 30 + classProb * 40  // Slow
        }
    }
    
    private fun storeRiskInFirestore(risk: Float) {
        val uid = auth.currentUser?.uid
        if (uid == null) {
            Log.e(TAG, "Cannot store risk: User not authenticated")
            return
        }
        
        val data = hashMapOf(
            "uid" to uid,
            "timestamp" to System.currentTimeMillis(),
            "risk" to risk
        )
        
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "STORING RISK: Attempting to store risk score ($risk) for user $uid")
                
                // Create a consistent document path for debugging
                val documentId = "risk_${System.currentTimeMillis()}"
                
                db.collection("risk_scores")
                    .document(uid)
                    .collection("transactions")
                    .document(documentId)  
                    .set(data)
                    
                Log.d(TAG, "STORING RISK: Successfully stored risk score in document: risk_scores/$uid/transactions/$documentId")
            } catch (e: Exception) {
                Log.e(TAG, "STORING RISK: Error storing risk score", e)
                
                // Check for network connectivity issues
                if (e.message?.contains("host", ignoreCase = true) == true || 
                    e.message?.contains("network", ignoreCase = true) == true ||
                    e.message?.contains("connect", ignoreCase = true) == true) {
                    Log.w(TAG, "STORING RISK: This appears to be a network connectivity issue. " +
                          "Check that your emulator/device has internet access.")
                }
                
                // Check for permission issues
                if (e.message?.contains("permission", ignoreCase = true) == true) {
                    Log.w(TAG, "STORING RISK: This appears to be a permissions issue. " +
                         "Verify your Firestore rules allow writing to this collection.")
                }
            }
        }
    }
    
    private fun storeRiskInFirestore(risk: Float, retryCount: Int = 0) {
        val uid = auth.currentUser?.uid ?: return
        val data = hashMapOf(
            "uid" to uid,
            "timestamp" to System.currentTimeMillis(),
            "risk" to risk
        )
        
        viewModelScope.launch(Dispatchers.IO) {
            try {
                db.collection("risk_scores")
                    .document(uid)
                    .collection("transactions")
                    .add(data)
                    .await()
                
                Log.d(TAG, "Successfully stored risk score")
            } catch (e: Exception) {
                Log.e(TAG, "Error storing risk score", e)
                
                // Implement retry logic with backoff
                if (retryCount < 3) {
                    delay(1000L * (retryCount + 1))  // Exponential backoff
                    storeRiskInFirestore(risk, retryCount + 1)
                    Log.d(TAG, "Retrying risk score storage (attempt ${retryCount + 1})")
                } else {
                    // Cache locally when retries fail
                    saveRiskScoreLocally(risk)
                }
            }
        }
    }
    
    private fun saveRiskScoreLocally(risk: Float) {
        try {
            val sharedPrefs = getApplication<Application>().getSharedPreferences(
                "cached_risk_scores", Context.MODE_PRIVATE)
            
            val cachedScores = sharedPrefs.getString("pending_scores", "")
            val newEntry = "${System.currentTimeMillis()}:$risk"
            val updatedScores = if (cachedScores.isNullOrEmpty()) newEntry else "$cachedScores,$newEntry"
            
            sharedPrefs.edit().putString("pending_scores", updatedScores).apply()
            Log.d(TAG, "Saved risk score locally for later upload")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cache risk score locally", e)
        }
    }
    
    fun resetAutoSignoutTimer() {
        autoSignoutJob?.cancel()
        autoSignoutJob = viewModelScope.launch {
            delay(AUTO_SIGNOUT_DELAY_MS)
            signOut()
        }
    }

    fun signOut() {
        viewModelScope.launch {
            try {
                // Store final risk score before signing out
                if (auth.currentUser != null && System.currentTimeMillis() - lastRiskStorageTime > 60000) {
                    // Only store if it's been at least a minute since last storage
                    storeRiskInFirestore(_riskScore.value)
                    Log.d(TAG, "Stored final risk score before sign out")
                }
                
                // Then proceed with sign out
                auth.signOut()
                // Rest of sign out code...
                
                // Clear any cached auth data from shared preferences or similar stores
                // (This should help prevent auto sign-in)
                getApplication<Application>().getSharedPreferences("auth_prefs", Context.MODE_PRIVATE)
                    .edit {
                        clear()
                    }
                    
                _authState.value = AuthState(isAuthenticated = false)
                _isSignedOut.value = true
                Log.d(TAG, "User signed out successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error signing out", e)
            }
        }
    }

    fun onAppBackground() {
        signOut()
        Log.d(TAG, "App went to background, signing out immediately")
    }

    fun onAppForeground() {
        Log.d(TAG, "App came to foreground")
    }

    private fun startAutoSignoutTimer() {
        resetAutoSignoutTimer()
        Log.d(TAG, "Started auto-signout timer")
    }

    override fun onCleared() {
        super.onCleared()
        sensorySession?.close()
        // Clear any active coroutines
        autoSignoutJob?.cancel()
        
        // Store final risk if needed
        if (auth.currentUser != null) {
            viewModelScope.launch(Dispatchers.IO) {
                storeRiskInFirestore(_riskScore.value)
                Log.d(TAG, "Stored final risk score during ViewModel cleanup")
            }
        }
    }

    class Factory(private val application: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(DashCamViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return DashCamViewModel(application) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}